all: figs
	make update

update:
	rsync -av figs/ --exclude *.png ~/Papers/nudging-topics/figs/

clean:
	-rm figs/*

figs/default.pdf: default.r
	Rscript $<

figs/supersize.pdf: supersize.r
	Rscript $<

figs/stoplight.pdf: stoplight.r
	Rscript $<

figs/default-exclude.pdf: default.r
	Rscript $< --exclude

figs/supersize-exclude.pdf: supersize.r
	Rscript $< --exclude

figs/stoplight-exclude.pdf: stoplight.r
	Rscript $< --exclude

figs: figs/default.pdf figs/supersize.pdf figs/stoplight.pdf figs/default-exclude.pdf figs/supersize-exclude.pdf figs/stoplight-exclude.pdf 