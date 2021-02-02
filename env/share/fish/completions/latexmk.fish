complete -c latexmk -x -a "(__fish_complete_suffix (commandline -ct) .tex '(La)TeX file')"
complete -c latexmk -o bibtex -d 'use bibtex when needed (default)'
complete -c latexmk -o bibtex- -d 'never use bibtex'
complete -c latexmk -o bibtex-cond -d 'use bibtex when needed, but only if the bib files exist'
complete -c latexmk -o bm -x -d 'Print message across the page when converting to postscript'
complete -c latexmk -o bi -x -d 'Set contrast or intensity of banner'
complete -c latexmk -o bs -x -d 'Set scale for banner'
complete -c latexmk -o commands -d 'list commands used by latexmk for processing files'
complete -c latexmk -o c -d 'clean up (remove) all nonessential files, except dvi, ps and pdf files'
complete -c latexmk -o C -o CA -d 'clean up (remove) all nonessential files'
complete -c latexmk -o CF -d 'Remove file of database of file information before doing other actions'
complete -c latexmk -o cd -d 'Change to directory of source file when processing it'
complete -c latexmk -o cd- -d 'Do NOT change to directory of source file when processing it'
complete -c latexmk -o dependents -o -deps -d 'Show list of dependent files after processing'
complete -c latexmk -o dependents- -o -deps- -d 'Do not show list of dependent files'
complete -c latexmk -o deps-out= -r -d 'Set name of output file for dependency list, and turn on showing of dependency list'
complete -c latexmk -o dF -x -d 'Filter to apply to dvi file'
complete -c latexmk -o dvi -d 'generate dvi'
complete -c latexmk -o dvi- -d 'turn off required dvi'
complete -c latexmk -o e -x -d 'Execute specified Perl code (as part of latexmk start-up code)'
complete -c latexmk -o f -d 'force continued processing past errors'
complete -c latexmk -o f- -d 'turn off forced continuing processing past errors'
complete -c latexmk -o gg -d 'Super go mode: clean out generated files before processing'
complete -c latexmk -o g -d 'process regardless of file timestamps'
complete -c latexmk -o g- -d 'Turn off -g'
complete -c latexmk -o h -o help -d 'print help'
complete -c latexmk -o jobname= -x -d 'set basename of output file(s) to STRING'
complete -c latexmk -o l -d 'force landscape mode'
complete -c latexmk -o l- -d 'turn off -l'
complete -c latexmk -o latex= -d 'set program used for latex' -xa '(__fish_complete_command)'
complete -c latexmk -o new-viewer -d 'in -pvc mode, always start a new viewer'
complete -c latexmk -o new-viewer- -d 'in -pvc mode, start a new viewer only if needed'
complete -c latexmk -o nobibtex -d 'never use bibtex'
complete -c latexmk -o nodependents -d 'Do not show list of dependent files after processing'
complete -c latexmk -o norc -d 'omit automatic reading of system, user and project rc files'
complete -c latexmk -o output-directory= -o outdir= -d 'set name of directory for output files' -xa '(__fish_complete_directories)'
complete -c latexmk -o pdf -d 'generate pdf by pdflatex'
complete -c latexmk -o pdfdvi -d 'generate pdf by dvipdf'
complete -c latexmk -o pdflatex= -d 'set program used for pdflatex' -xa '(__fish_complete_command)'
complete -c latexmk -o pdfps -d 'generate pdf by ps2pdf'
complete -c latexmk -o pdf- -d 'turn off pdf'
complete -c latexmk -o ps -d 'generate postscript'
complete -c latexmk -o ps- -d 'turn off postscript'
complete -c latexmk -o pF -x -d 'Filter to apply to postscript file'
complete -c latexmk -o p -d 'print document after generating postscript'
complete -c latexmk -o print=dvi -d 'when file is to be printed, print the dvi file'
complete -c latexmk -o print=ps -d 'when file is to be printed, print the ps file (default)'
complete -c latexmk -o print=pdf -d 'when file is to be printed, print the pdf file'
complete -c latexmk -o pv -d 'preview document'
complete -c latexmk -o pv- -d 'turn off preview mode'
complete -c latexmk -o pvc -d 'preview document and continuously update'
complete -c latexmk -o pvc- -d 'turn off -pvc'
complete -c latexmk -o quiet -d 'silence progress messages from called programs'
complete -c latexmk -o r -r -d 'Read custom RC file'
complete -c latexmk -o recorder -d 'Use -recorder option for (pdf)latex'
complete -c latexmk -o recorder- -d 'Do not use -recorder option for (pdf)latex'
complete -c latexmk -o rules -d 'Show list of rules after processing'
complete -c latexmk -o rules- -d 'Do not show list of rules after processing'
complete -c latexmk -o silent -d 'silence progress messages from called programs'
complete -c latexmk -o time -d 'show CPU time used'
complete -c latexmk -o time- -d 'don\'t show CPU time used'
complete -c latexmk -o use-make -d 'use the make program to try to make missing files'
complete -c latexmk -o use-make- -d 'don\'t use the make program to try to make missing files'
complete -c latexmk -o v -d 'display program version'
complete -c latexmk -o verbose -d 'display usual progress messages from called programs'
complete -c latexmk -o version -d 'display program version'
complete -c latexmk -o view=default -d 'viewer is default (dvi, ps, pdf)'
complete -c latexmk -o view=dvi -d 'viewer is for dvi'
complete -c latexmk -o view=none -d 'no viewer is used'
complete -c latexmk -o view=ps -d 'viewer is for ps'
complete -c latexmk -o view=pdf -d 'viewer is for pdf'

