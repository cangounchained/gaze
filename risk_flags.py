def get_gaze_flags(fixation_counter, fixation_durations, session_time):
    total_fixations = sum(fixation_counter.values()) or 1
    eye_pct = (fixation_counter['eyes_left'] + fixation_counter['eyes_right']) / total_fixations
    mouth_pct = fixation_counter['mouth'] / total_fixations
    none_pct = fixation_counter['none'] / total_fixations

    # Count gaze switches
    switches = 0
    last_target = None
    for entry in fixation_durations:
        if entry['target'] != last_target:
            switches += 1
            last_target = entry['target']

    flags = []
    if eye_pct > 0.9:
        flags.append("[!] Excessive eye fixation (>90%)")
    if mouth_pct < 0.05:
        flags.append("[!] Very little attention to mouth (<5%)")
    if none_pct > 0.2:
        flags.append("[!] High unfocused gaze (>20%)")
    if switches < 3:
        flags.append("[!] Very few gaze switches (<3 transitions)")

    return flags
