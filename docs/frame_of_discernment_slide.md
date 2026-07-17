# Frame of Discernment Slide Deck

Some pull-request systems reject binary files, so the generated PowerPoint deck
is not stored directly in the repository. Generate the downloadable `.pptx` file
locally with:

```bash
python scripts/generate_frame_of_discernment_pptx.py
```

By default, this writes:

```text
docs/frame_of_discernment_production.pptx
```

You can also pass a custom output path:

```bash
python scripts/generate_frame_of_discernment_pptx.py /tmp/frame_of_discernment_production.pptx
```

The generated slide summarizes how this project produces the frame of
discernment Θ: configured hypotheses are encoded as focal-element bit masks,
small frames use all non-empty subsets, larger frames use singleton masks,
inferred aircraft-type groups, and one full-frame uncertainty mask.
