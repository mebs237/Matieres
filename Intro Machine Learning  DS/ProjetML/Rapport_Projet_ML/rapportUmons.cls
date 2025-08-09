\ProvidesClass{rapportUmons}[2023/11/21 Class pour les rapports à l'Umons]
\LoadClass[16pt,a4paper]{article}

% Encoding for French
\usepackage[T1]{fontenc}
% French language support
\usepackage[french]{babel}


% Formatting and font
\usepackage[paperwidth=21.0cm,paperheight=29.7cm,top=2cm,left=2.cm,right=2.cm,bottom=3.cm]{geometry}
\parskip=0.5\baselineskip
\parindent=0pt
\RequirePackage{pxfonts}

\RequirePackage[pdftex,%
	hyperindex=true,
	colorlinks=true,
	linkcolor=navy,
	anchorcolor=magenta,
	citecolor=navy,
	urlcolor=navy,
	unicode,
	implicit=true,
	pdfborder={0 0 0}]{hyperref}

\RequirePackage{color}
\RequirePackage{tcolorbox}
\definecolor{navy}{rgb}{0,0,0.4}
\definecolor{mygreen}{RGB}{0,128,0}
\usepackage{listingsutf8}

\lstset{
    inputencoding=utf8/latin1, % Encodage du fichier source
    extendedchars=true,
    literate=%
        {é}{{\'e}}1 {è}{{\`e}}1 {ê}{{\^e}}1
        {à}{{\`a}}1 {ä}{{\"a}}1 {â}{{\^a}}1
        {ç}{{\c{c}}}1 {œ}{{\oe}}1 {ë}{{\"e}}1
        {ï}{{\"i}}1 {û}{{\^u}}1 {ü}{{\"u}}1
        {ô}{{\^o}}1 {ö}{{\"o}}1 {ù}{{\`u}}1
        {É}{{\'E}}1 {È}{{\`E}}1 {Ê}{{\^E}}1
        {À}{{\`A}}1 {Ä}{{\"A}}1 {Â}{{\^A}}1
        {Ç}{{\c{C}}}1 {Œ}{{\OE}}1 {Ë}{{\"E}}1
        {Ï}{{\"I}}1 {Û}{{\^U}}1 {Ü}{{\"U}}1
        {_}{{\_}}1 {\\}{{\textbackslash}}1
        {<}{{$<$}}1 {>}{{$>$}}1
        {{[}}{{$\lbrace$}}1 {]}{{$\rbrace$}}1
        {{\{}}{{$\lbrace$}}1 {\}}{{$\rbrace$}}1
        {||}{{$\parallel$}}1  {^}{{$\wedge$}}1
         {Ö}{{\"O}}1 {Ù}{{\`U}}1,
    frame=single,
    language=Octave,
    breaklines=true,
    escapechar=\,
    breakatwhitespace=true,
    basicstyle=\footnotesize\ttfamily,
    numbers=left,
    stepnumber=1,
    numberstyle=\tiny\color{gray},
    captionpos=b,
    keywordstyle=\color{blue},
    commentstyle=\color{mygreen},
    stringstyle=\color{purple},
      }

\usepackage[utf8x]{inputenc} 
\usepackage{libertine}
%\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{graphicx} % Required for inserting images
\usepackage{pdfpages}
\usepackage{appendix}
\usepackage{indentfirst}
\setlength{\parindent}{0.5cm}