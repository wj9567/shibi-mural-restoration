"""Stage 6: separated identification and candidate-review states, with audit history.

Two independent decisions are tracked per defect region:

* identification  — is the detected region real and is its mask usable?
* candidate review — may one specific repair result leave the app as an output?

Every manual change bumps the region revision and marks older candidates and AI
assessments stale, so a review of an old mask can never authorise new pixels.
All decisions append an immutable review_event record.
"""
from .storage import new_id, timestamp

IDENTIFICATION_STATES = {'pending', 'confirmed', 'expert_review'}
CANDIDATE_STATES = {'accepted', 'accepted_hypothesis', 'rejected', 'expert_review', 'experimental_preview'}
STALE_REASONS = {
    'mask': '修复范围或区域已人工修改',
    'recommendation': '修复建议已人工修改',
    'removed': '病害区域已删除',
}


def revision(proposal):
    return int((proposal or {}).get('revision') or 1)


def identification_state(proposal):
    """Current identification decision, defaulting from the legacy confirmed flag."""
    if not proposal:
        return 'pending'
    status = ((proposal.get('identification') or {}).get('status') or '').strip()
    if status in IDENTIFICATION_STATES:
        return status
    return 'confirmed' if proposal.get('confirmed') else 'pending'


def candidate_state(candidate):
    review = (candidate or {}).get('review') or {}
    status = (review.get('status') or '').strip()
    return status if status in CANDIDATE_STATES else 'pending'


def record_event(store, *, image_id, proposal_id, action, changes=None, decision='', note=''):
    event = {
        'id': new_id(), 'image_id': image_id, 'proposal_id': proposal_id or '',
        'action': action, 'created_at': timestamp(), 'changes': changes or {},
        'decision': decision or '', 'note': str(note or '')[:2000],
    }
    store.put('review_event', event)
    return event


def list_events(store, image_id, proposal_id=''):
    events = [event for event in store.list('review_event')
              if event.get('image_id') == image_id
              and (not proposal_id or event.get('proposal_id') == proposal_id)]
    events.sort(key=lambda event: event.get('created_at') or '')
    return events


def set_identification(store, proposal, status, note=''):
    if status not in IDENTIFICATION_STATES:
        raise ValueError('识别状态无效。')
    state = {
        'status': status, 'note': str(note or '')[:2000], 'time': timestamp(),
    }
    # `confirmed` is kept for compatibility with the older repair gates.
    confirmed = status in {'confirmed', 'expert_review'}
    return store.update('proposal', proposal['id'], identification=state, confirmed=confirmed)


def set_candidate_review(store, job_id, candidate_id, status, note='', decision=''):
    if status not in CANDIDATE_STATES:
        raise ValueError('候选审核状态无效。')
    job = store.get('job', job_id)
    if not job:
        raise ValueError('修复任务不存在。')
    candidates = []
    updated = None
    for candidate in job.get('candidates') or []:
        if candidate['id'] == candidate_id:
            updated = {**candidate, 'review': {'status': status, 'note': str(note or '')[:2000],
                                               'time': timestamp(), 'decision': decision}}
            candidates.append(updated)
        else:
            candidates.append(candidate)
    if updated is None:
        raise ValueError('候选不属于当前任务。')
    store.update('job', job_id, candidates=candidates)
    return updated


def candidate_stale(store, job, candidate):
    """True when the candidate was produced from an older region revision."""
    if candidate.get('stale'):
        return True, candidate.get('stale_reason') or '候选已过期'
    proposal = store.get('proposal', job.get('defect_proposal_id')) if job.get('defect_proposal_id') else None
    snapshot = candidate.get('proposal_revision')
    if proposal and snapshot is not None and int(snapshot) != revision(proposal):
        return True, '区域已更新，候选基于旧修复范围或旧修复建议'
    return False, ''


def mark_stale(store, proposal_id, reason):
    """Invalidate candidates, AI reviews and assessments that used the old region."""
    touched = 0
    stamp = timestamp()
    for job in store.list('job'):
        if job.get('defect_proposal_id') != proposal_id:
            continue
        candidates = []
        changed = False
        for candidate in job.get('candidates') or []:
            if not candidate.get('stale'):
                candidate = {**candidate, 'stale': True, 'stale_reason': reason, 'stale_at': stamp}
                changed = True
            candidates.append(candidate)
        fields = {}
        if changed:
            fields['candidates'] = candidates
        review = job.get('ai_review')
        if review and not review.get('stale'):
            fields['ai_review'] = {**review, 'stale': True, 'stale_reason': reason, 'stale_at': stamp}
        if fields:
            store.update('job', job['id'], **fields)
            touched += 1
    for assessment in store.list('assessment'):
        if proposal_id in (assessment.get('region_ids') or []) and not assessment.get('stale'):
            store.update('assessment', assessment['id'], stale=True,
                         stale_reason=reason, stale_at=stamp)
            touched += 1
    return touched


def bump_revision(store, proposal_id, reason, changes=None, action='update', note=''):
    proposal = store.get('proposal', proposal_id)
    if not proposal:
        raise ValueError('缺损记录不存在。')
    next_revision = revision(proposal) + 1
    store.update('proposal', proposal_id, revision=next_revision)
    stale = mark_stale(store, proposal_id, reason)
    record_event(store, image_id=proposal['image_id'], proposal_id=proposal_id, action=action,
                 changes={'revision': next_revision, **(changes or {})}, note=note)
    return next_revision, stale


def review_candidate(store, job, candidate_id, action, note='', archive_as_hypothesis=False):
    """Apply one candidate decision; returns ``(job, candidate, event)``."""
    from .professional import validate_candidate_selection
    if job.get('status') in {'queued', 'planning', 'running', 'evaluating'}:
        raise ValueError('任务完成前不能审核候选。')
    candidate = next((item for item in job.get('candidates') or [] if item['id'] == candidate_id), None)
    if candidate is None:
        raise ValueError('候选不属于当前任务。')
    stale, reason = candidate_stale(store, job, candidate)
    if stale:
        raise ValueError(f'该候选已过期（{reason}），请重新生成局部修复候选后再审核。')

    proposal = store.get('proposal', job.get('defect_proposal_id')) if job.get('defect_proposal_id') else None
    state = identification_state(proposal)
    if action in {'accept', 'reject', 'expert_review'} and proposal and state == 'pending':
        raise ValueError('请先确认病害识别，再审核修复候选。')

    if action == 'accept':
        selection = validate_candidate_selection(
            job, candidate, archive_as_hypothesis=archive_as_hypothesis, decision='adopt')
        status = 'accepted_hypothesis' if selection['selected_as'] == 'hypothesis' else 'accepted'
        job = store.update('job', job['id'], selected_id=candidate_id, selected_as=selection['selected_as'],
                           review={'note': str(note or '')[:2000], 'time': timestamp(),
                                   'archive_as_hypothesis': selection['archive_as_hypothesis'],
                                   'decision': 'adopt'})
        candidate = set_candidate_review(store, job['id'], candidate_id, status, note, decision='adopt')
        if proposal and state == 'expert_review':
            # An explicit human acceptance resolves a pending expert flag and is recorded.
            set_identification(store, proposal, 'confirmed', note)
    elif action == 'reject':
        validate_candidate_selection(job, candidate, decision='reject')
        job = store.update('job', job['id'], selected_id=candidate_id, selected_as='rejected',
                           review={'note': str(note or '')[:2000], 'time': timestamp(),
                                   'archive_as_hypothesis': False, 'decision': 'reject'})
        candidate = set_candidate_review(store, job['id'], candidate_id, 'rejected', note, decision='reject')
    elif action == 'expert_review':
        if not proposal:
            raise ValueError('只有病害区域的候选可以标记专家复核。')
        set_identification(store, proposal, 'expert_review', note)
        job = store.update('job', job['id'], selected_id=candidate_id,
                           selected_as='expert_review',
                           review={'note': str(note or '')[:2000], 'time': timestamp(),
                                   'archive_as_hypothesis': False, 'decision': 'expert_review'})
        candidate = set_candidate_review(store, job['id'], candidate_id, 'expert_review', note,
                                         decision='expert_review')
    elif action == 'experimental_preview':
        risk = job.get('defect_risk_class') or (proposal or {}).get('region_class') or 'uncertain'
        if not (candidate.get('speculative') or risk in {'high', 'unrestorable'}):
            raise ValueError('试验性预览只用于高风险或推测性候选；普通候选请直接审核。')
        job = store.update('job', job['id'], preview_candidate_id=candidate_id)
        candidate = set_candidate_review(store, job['id'], candidate_id, 'experimental_preview', note,
                                         decision='preview')
    else:
        raise ValueError('不支持的候选审核操作。')

    event = record_event(
        store, image_id=job['image_id'], proposal_id=(proposal or {}).get('id', ''),
        action=f'candidate_{action}',
        changes={'candidate_id': candidate_id, 'candidate_status': candidate['review']['status'],
                 'selected_as': job.get('selected_as'),
                 'proposal_revision': revision(proposal) if proposal else None},
        decision=candidate['review'].get('decision', ''), note=note)
    return store.get('job', job['id']), candidate, event


def export_guard(store, job, candidate):
    """Region-aware export gate; falls back to the legacy rule for manual jobs."""
    from .professional import validate_selected_export
    selection = validate_selected_export(job, candidate)
    proposal = store.get('proposal', job.get('defect_proposal_id')) if job.get('defect_proposal_id') else None
    if not proposal:
        return selection
    state = identification_state(proposal)
    if state == 'pending':
        raise ValueError('病害识别尚未确认，不能导出正式修复结果。')
    if state == 'expert_review':
        raise ValueError('该区域已标记需要专家复核，专家结论录入前不能导出正式结果。')
    stale, reason = candidate_stale(store, job, candidate)
    if stale:
        raise ValueError(f'候选已过期（{reason}），请重新生成局部修复候选后再导出。')
    status = candidate_state(candidate)
    if status == 'rejected':
        raise ValueError('该候选已被驳回，禁止导出或保存修复成图。')
    if status == 'experimental_preview':
        raise ValueError('试验性预览结果只是推测性参考，不能作为正式修复结果导出。')
    if status == 'expert_review':
        raise ValueError('该候选已标记需要专家复核，复核完成前不能导出。')
    return selection
