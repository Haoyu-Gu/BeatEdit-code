"""Region-bounded TagFill completion with lengths predicted from a blank gap."""


def generate_gap(pipeline, tokens, start, end, track, parser, config, build_skeleton,
                 inserter_steps=2, confidence_threshold=0.5, temperature=1.0,
                 top_k=0, harmony_weight=0.0):
    if track not in (0, 1, 2) or start < 0 or end <= start:
        raise ValueError('Use a nonempty beat range and mask_track 0, 1, or 2')
    if harmony_weight:
        raise ValueError('Gap completion preserves context; use harmony_weight=0')
    if inserter_steps < 1 or temperature <= 0 or top_k < 0:
        raise ValueError('Invalid inserter decoding parameters')
    tokens = list(tokens)
    parsed = parser.parse_sequence(tokens)
    tracks = parser.separate_tracks(parsed)
    selected = []
    for track_id in ((0, 1) if track == 2 else (track,)):
        if end > len(tracks[track_id]):
            raise ValueError('Gap extends beyond the available beat boundaries')
        for beat in tracks[track_id][start:end]:
            selected.append((beat['start_idx'], beat['end_idx'], track_id))
    selected.sort()

    # Remove content BEFORE tagging: the old/target token count is never an input.
    blank = []
    spans = []
    cursor = 0
    has_tracks = hasattr(config, 'TRACK0_START')
    bundled = hasattr(config, 'SPLIT_0')
    for left, right, track_id in selected:
        blank.extend(tokens[cursor:left])
        empty = ([config.TRACK0_START if track_id == 0 else config.TRACK1_START, 0]
                 if has_tracks else [config.EMPTY_MARKER])
        spans.append((len(blank), len(blank) + len(empty), track_id))
        blank.extend(empty)
        cursor = right
    blank.extend(tokens[cursor:])
    if len(blank) > pipeline.tagger.config.max_position_embeddings:
        raise ValueError('Blank-gap input exceeds the tagger context window')
    labels = pipeline._run_tagger(blank)
    if len(labels) != len(blank):
        raise ValueError('Tagger must return one label per input token')

    # Only empty-beat anchors may allocate content. Everything else is immutable.
    skeleton = []
    output_spans = []
    positions = []
    cursor = 0
    for left, right, track_id in spans:
        skeleton.extend(blank[cursor:left])
        beat_labels = [config.LABEL_KEEP] * (right - left)
        op, _ = config.decode_felix_label(labels[right - 1])
        if op == 'APPEND':
            beat_labels[-1] = labels[right - 1]
        built = build_skeleton(blank[left:right], beat_labels, [[] for _ in beat_labels])
        content = built['skeleton_tokens']
        # Keep the empty anchor during filling, as in training skeletons.
        # Convert it to a nonempty beat marker only after the fill step.
        positions.extend(len(skeleton) + pos for pos in built['mask_positions'])
        output_spans.append((len(skeleton), len(skeleton) + len(content), track_id))
        skeleton.extend(content)
        cursor = right
    skeleton.extend(blank[cursor:])
    if len(skeleton) > pipeline.inserter.config.max_position_embeddings:
        raise ValueError('Predicted gap skeleton exceeds the inserter context window')
    if positions:
        if inserter_steps > 1:
            filled = pipeline._run_inserter_iterative(
                skeleton, positions, temperature, top_k, inserter_steps, confidence_threshold)
        else:
            filled = pipeline._run_inserter(skeleton, positions, temperature, top_k)
    else:
        filled = skeleton
    if len(filled) != len(skeleton):
        raise ValueError('Inserter changed the skeleton length')

    # Normalize only generated beats; copy every context token from the input.
    result = []
    cursor = 0
    for left, right, track_id in output_spans:
        result.extend(skeleton[cursor:left])
        generated = filled[left:right]
        content = generated[2:] if has_tracks else generated[1:]
        notes = parser.decode_beat(content)
        encoded = parser.encode_beat(notes)
        if has_tracks:
            marker = config.TRACK0_START if track_id == 0 else config.TRACK1_START
            result.extend([marker] + encoded)
        elif bundled:
            marker = config.SPLIT_0 if track_id == 0 else config.SPLIT_1
            result.extend([marker] + encoded if notes else [config.EMPTY_MARKER])
        else:
            result.extend(encoded)
        cursor = right
    result.extend(skeleton[cursor:])
    return result
