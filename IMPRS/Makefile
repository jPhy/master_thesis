presentation.pdf : %.pdf : %.tex clean
	pdflatex --draftmode $< ; pdflatex --draftmode $< ; pdflatex --draftmode $< ; pdflatex $<

.PHONY : clean
clean:
	find . -maxdepth 1 -type f -and \( -not -name '*.tex' \) -and \( -not -name 'Makefile' \) -and \( -not -name '.gitignore' \) -delete
