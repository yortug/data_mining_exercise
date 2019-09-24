
--------------------------------------------------------------------------------

AUTHOR: TROY ENGELHARDT
STUDENT ID: 18815179
DATE: 2019-09-24

--------------------------------------------------------------------------------

FOR YOUR CONVENIENCE THE KNITTED JUPYTER NOTEBOOK IS BEING HOSTED ONLINE HERE:

https://yortug.github.io/data_mining_assignment/data_mining_assignment.html

AND THE FULL REPOSITORY IS BEING HOSTED HERE:

https://github.com/yortug/data_mining_exercise

--------------------------------------------------------------------------------

PRELUDE

--------------------------------------------------------------------------------


THE FOLLOWING IS A TREE-STRUCTURE OVERVIEW OF THIS PROJECT DIRECTORY. THE MAIN
FILES TO CONSIDER ARE THE ANALYSIS.PY, PREDICT.CSV AND REPORT.PDF FILES. THE
ANALYSIS.PY FILE ONLY HAS NUMPY, PANDAS AND SKLEARN DEPENDENCIES. THE OTHER
LARGER NOTEBOOK ANALYSIS FILES HAVE EXTERNAL DEPENDENCIES, BUT THEIR RESULTS
CAN BE SEEN IN THE KNITTED .HTML OR .PDF FILES AS YOU PLEASE. THE REPORT.PDF
FILE CONTAINS THE OVERALL WRITE-UP OF THE ASSIGNMENT, IT SHOULD BE OF PRIMARY
FOCUS, SUPPLEMENTED BY THE ANALYSIS.PY FILE, AND THEN OVERALL ACCURACY CAN BE
MEASURED USING THE PREDICT.CSV FILE. ALL OTHER FILES ARE SUPPLEMENTARY.

.
├── README.txt
├── analysis.py
├── assignment_resources
│   └── assignment2019.pdf
├── comp3009_assignment_notebook.ipynb
├── data2019.student.csv
├── declaration_of_originality\ [signed].pdf
├── notebooks
│   ├── comp3009_assignment_notebook.html
│   └── comp3009_assignment_notebook.pdf
├── predict.csv
├── report.pdf
├── train_test_val
│   ├── arff
│   │   ├── test.arff
│   │   ├── train.arff
│   │   └── val.arff
│   ├── csv
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── val.csv
│   └── final_100_set.csv
└── visualisations
    ├── 1.\ boxplot_att25.html
    ├── 2.\ boxplot_att28.html
    ├── 3.\ boxplot_att20_numeric.html
    ├── 4.\ boxplot_att12_categorical.html
    ├── 5.\ hist_att21_before_scale.html
    ├── 6.\ hist_att21_after_scale.html
    ├── 7.\ hbar_univariate_feature_importance.html
    ├── 8.\ hbar_tree_feature_importance.html
    └── 9.\ boxplot_classification_comparison.html

--------------------------------------------------------------------------------

MORE INFORMATION

--------------------------------------------------------------------------------

EXAMPLE VISUALISATIONS FROM THE ANALYSIS CAN BE FOUND IN THE VISUALISATIONS
DIRECTORY. FAR MORE WERE GENERATED DURING ANALYSIS USING INTERACTIVE WIDGETS,
THE CODE FOR WHICH CAN BE FOUND COMMENTED OUT IN THE ACCOMPANYING NOTEBOOKS.

THE TRAINING, TESTING AND VALIDATION SETS CAN BE FOUND WITHIN THE TRAIN_TEST
_VAL DIRECTORY, CONTAINING SUB-DIRECTORIES FOR BOTH ARFF AND CSV FORMATS. THE
ARFF FILES WERE CONVERTED FROM THEIR PANDAS DATA FRAMES USING A FUNCTION FOUND
ONLINE, HOSTED ON GITHUB UNDER THE MIT LICENCE.

https://github.com/saurabhnagrecha/Pandas-to-ARFF/blob/master/pandas2arff.py

THE ONLY DIRECTORY WHICH HASN'T BEEN MENTIONED IS THE NOTEBOOKS DIRECTORY.
THIS FOLDER CONTAINS TWO KNITTED VERSIONS OF THE NOTEBOOK USED FOR ANALYSIS,
FORMATTED TO PDF AND HTML FORMAT FOR YOUR CONVENIENCE. THIS IS DONE SO YOU
DON'T HAVE TO RUN THE OVERALL NOTEBOOK ON A LOCAL SERVER JUST TO SEE THE
ANALYSIS/OUTPUT GENERATED THERE.

--------------------------------------------------------------------------------

INSTRUCTIONS TO RUN THE CODE

--------------------------------------------------------------------------------

THE BEST WAY TO RUN THE CODE IS AS FOLLOWS:
  1. OPEN YOUR TERMINAL
  2. NAVIGATE INTO THE PROJECT DIRECTORY
  3. ENTER INTO TERMINAL:
        > python analysis.py
  4. WAIT APPROXIMATELY 10 TO 30 SECONDS FOR EXECUTION TO COMPLETE
  5. OBSERVE THE STRING RESULTS OUTPUT TO YOUR TERMINAL
        I. TABLE OF ACC/F1 (DURING CV) FOR ALL TESTED MODELS
        II. TABLE OF PROPORTION OF PREDICTED 0/1'S ON THE FINAL 100 ROWS
        III. TABLE OF THE SET OF PREDICTIONS OF THE FINAL 100 ROWS

NOTE: EXECUTING THIS CODE WILL NOT OUTPUT ANY FILES. THIS IS BY DESIGN. DUE TO
THE RANDOM ELEMENTS OF SPECIFIC MODELS, PREDICTIONS CAN CHANGE. YOU CAN, IF YOU
WANT, ADD THE .TO_CSV('FILENAME') METHOD TO THE END OF THE OUTPUT FILES TO TEST
THEM YOURSELF. HOWEVER, IT CAN CLEARLY BE SEEN WITHIN THE CODE, AND WITHIN THE
NOTEBOOKS THAT THE PREDICTIONS FILE PRESENT HAS BEEN GENERATED FROM THE SETS OF
MODELS WHICH WERE PROVIDED.

--------------------------------------------------------------------------------

ALTERNATIVE INSTRUCTIONS

--------------------------------------------------------------------------------

YOU CAN RUN THE FOLLOWING COMMAND WHILE INSIDE THE MAIN DIRECTORY TO RUN THE
JUPYTER NOTEBOOK SERVER AND DYNAMICALLY VIEW THE IPYNB FILE.

    > jupyter notebook

THEN YOU CAN SIMPLY SELECT `CELLS` AT THE TOP, AND CLICK `RUN ALL` TO RUN ALL
THE CELLS WITHIN THE NOTEBOOK. YOU CAN THEN SCROLL THROUGH THE DOCUMENT AND
MAKE CHANGES WHERE EVER YOU PLEASE.

HOWEVER, THIS OF COURSE HAS A DEPENDENCY ON THE JUPYTER PACKAGE. AN ALTERNATIVE
IS TO SIMPLY VIEW THE EXPORTED HTML AND PDF FILES WITHIN THE NOTEBOOKS FOLDER.
THIS DOES NOT REQUIRE THE WEB SERVER AND ANALYTICS PLATFORM TO BE RUN LOCALLY.


--------------------------------------------------------------------------------

CLOSING

--------------------------------------------------------------------------------

IF YOU HAVE ANY PROBLEMS RUNNING THE FILES, PLEASE DON'T HESITATE TO CONTACT ME
AT 18815179@STUDENT.CURTIN.EDU.AU! THESE INSTRUCTIONS HOPEFULLY SUFFICE.

--------------------------------------------------------------------------------
