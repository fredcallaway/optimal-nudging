all: figs
	make update

update:
	rsync -av figs/ --exclude '*.png' ~/Papers/nudging-topics/figs/results/
	rsync -av stats/ ~/Papers/nudging-topics/stats/

clean:
	-rm figs/*

figs/default.pdf: default.r base.r
	Rscript $<

figs/supersize.pdf: supersize.r base.r
	Rscript $<

figs/stoplight.pdf: stoplight.r base.r
	Rscript $<

figs/default-full.pdf: default.r base.r
	Rscript $< --full

figs/supersize-full.pdf: supersize.r base.r
	Rscript $< --full

figs/stoplight-full.pdf: stoplight.r base.r
	Rscript $< --full

figs/cost_reduction.pdf: cost_reduction.r base.r
	Rscript $<

figs/belief_modification.pdf: belief_modification.r base.r
	Rscript $<

figs: \
	figs/default.pdf figs/supersize.pdf figs/stoplight.pdf \
	figs/default-full.pdf figs/supersize-full.pdf figs/stoplight-full.pdf \
# 	figs/cost_reduction.pdf figs/belief_modification.pdf
