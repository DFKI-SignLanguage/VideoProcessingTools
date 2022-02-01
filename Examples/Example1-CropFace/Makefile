# Takes all the videos in the INDIR and outputs the cropped face in OUTDIR.
# Intermediate .bounds files are also stored in the OUTDIR

INDIR=videos
OUTDIR=cropped-faces

# List of all MP4 videos
invideofiles := $(wildcard $(INDIR)/*.mp4)

# Compose the names of output video files and intermediate 'bounds'.
outvideofiles := $(subst $(INDIR)/,$(OUTDIR)/,$(invideofiles))
boundfiles := $(subst .mp4,.bounds,$(outvideofiles))


all: $(outvideofiles)
	@echo "Cropped the face of video files..."

$(outvideofiles): $(OUTDIR)/%.mp4: $(OUTDIR)/%.bounds $(INDIR)/%.mp4
	@echo "Cropping face from video $(word 2, $^) to video $@ using bounds $(word 1, $^)"
	PYTHONPATH=../.. python -m slvideotools.crop_video --invideo $(word 2, $^) --outvideo $@ --inbounds $(word 1, $^)

$(boundfiles): $(OUTDIR)/%.bounds: $(INDIR)/%.mp4 | $(OUTDIR)
	@echo "Finding bounds for video $<, saving to $@"
	PYTHONPATH=../.. python -m slvideotools.extract_face_bounds --invideo $< --outbounds $@

$(OUTDIR):
	mkdir -p $@

clean:
	rm -rf $(OUTDIR)
