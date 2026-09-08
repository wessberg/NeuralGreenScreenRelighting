# Music-video organization

This workspace is intended for multiple independent music videos. Each video may contain footage of the user's performance from multiple angles and many individually directed scenes.

Suggested structure for new productions:

```text
music_videos/<video_id>/
  project.json                 # song, creative brief, shot catalog and delivery
  audio/                       # untouched master/stems/reference audio
  lyrics/                      # text and optional LRC/SRT/JSON timing
  analysis/                    # onsets, beat grid, section/lyric cues, confidence
  lighting/                    # artistic cue definitions in song time
  shots/<shot_id>/
    shot.json                  # source metadata, angle, sync offset, shot direction
    source/ key/ blender/ relight/ composite/ intermediate/ final/ logs/
  edit/                        # ordered shot ranges, transitions and timeline
  final/                       # assembled music video HDR/SDR masters and QC
scene_library/<scene_id>/       # optional reusable native Blender environments
scripts/                       # reusable automation, currently example-tuned
```

Identifiers and explicit references determine ownership; do not guess which song or video a new clip belongs to from its filename. A scene can be reused by different shots, with independently matched cameras and lighting variants. Keep shot data separate from reusable environment definitions.

Synchronization convention: `music_time = source_frame_index / source_fps + music_time_at_source_frame_zero_seconds`, subject to any separately documented measured clock drift. A selected range adds its source start-frame offset. Store the frame rate as a rational value. Blender lights and actor-relighting conditions must evaluate identical cues at identical music times.

The current proof-of-concept files remain at the workspace root; `proof_of_concept/project.json` points there. The existing scripts are still tuned for this example, as stated in the root README. This layout and the brief templates are the architecture for generalizing the pipeline, not a claim that a complete multi-shot editor already exists.
