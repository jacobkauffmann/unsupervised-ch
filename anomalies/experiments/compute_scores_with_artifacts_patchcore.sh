#!/usr/bin/env sh

categories=(bottle capsule pill toothbrush wood)
artifact=(original salt\&pepper blurring sharpening checkerboard)
corrected=(False True)

for cat in ${categories[@]}; do
    for art in ${artifact[@]}; do
        for cor in ${corrected[@]}; do
            if [ $art == "original" ] && [ $cor == "True" ]; then
                continue
            fi
            python -m experiments.compute_scores_with_artifacts_patchcore --category $cat --artifact $art --corrected $cor
        done
    done
done

exit 0
