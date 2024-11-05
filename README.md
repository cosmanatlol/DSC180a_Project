# DSC180a_Project
This is the accompanying code regarding the analysis of the Electric Vehicle sector. It includes a Jupyter notebook showing preprocessing and all tools used in the analysis.
## How To Use
To see the EDA and technical analyis select the "example.ipynb" and install packages if required. Then run all cells. If you want to just use our preprocessing tools, call the function "preprocess" in preprocessing.py with a list of tickers, start date, and end date. It will return a dictionary with a DataFrame for each ticker that was selected. If train = True it will append a target variable "Target" which is a boolean if the stock increased  in price for the next day. This preprocessing tool is particularly useful for tree boosting/ensemble models which we show an example in "example.ipynb" (NOT COMPLETED). Lastly there a tiny time mixer model initialized in "TTM.ipynb" which predicts the next days closing value which can also be used as an imput in a tree model.