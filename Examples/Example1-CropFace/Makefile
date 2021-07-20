# Takes all the videos in the INDIR and outputs the cropped face in OUTDIR.
# Intermediate .bounds files are also stored in the OUTDIR

#INDIR="/Users/fanu01-admin/Nextcloud/Shared/ACG Material/Projects/Active/EASIER/WP7/EkmanClassifierSharedData/FePh/FePh_images"
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
	PYTHONPATH=../.. python -m dfki_sl_videotools.crop_video --invideo $(word 2, $^) --outvideo $@ --inbounds $(word 1, $^)

$(boundfiles): $(OUTDIR)/%.bounds: $(INDIR)/%.mp4 | $(OUTDIR)
	@echo "Finding bounds for video $<, saving to $@"
	PYTHONPATH=../.. python -m dfki_sl_videotools.extract_face_bounds --invideo $< --outbounds $@

$(OUTDIR):
	mkdir -p $@
