all: figs
	make update

update:
	rsync -av figs/ --exclude *.png ~/Papers/nudging-topics/figs/
	rsync -av stats/ ~/Papers/nudging-topics/stats/

clean:
	-rm figs/*

figs/default.pdf: default.r base.r
	Rscript $<

figs/supersize.pdf: supersize.r base.r
	Rscript $<

figs/stoplight.pdf: stoplight.r base.r
	Rscript $<

figs/default-exclude.pdf: default.r base.r
	Rscript $< --exclude

figs/supersize-exclude.pdf: supersize.r base.r
	Rscript $< --exclude

figs/stoplight-exclude.pdf: stoplight.r base.r
	Rscript $< --exclude

figs/cost_reduction.pdf: cost_reduction.r base.r
	Rscript $<

figs/belief_modification.pdf: belief_modification.r base.r
	Rscript $<

figs: figs/default.pdf figs/supersize.pdf figs/stoplight.pdf figs/default-exclude.pdf figs/supersize-exclude.pdf figs/stoplight-exclude.pdf figs/cost_reduction.pdf figs/belief_modification.pdf
