**Project Title:** New York Times Front Page Article Detection

**Project Description:**
	
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The goal of this project is to see if key characteristics of an article can be used to detect whether the article will be on the front page of the New York Times (NYT). Data was scraped from the NYT API in the nytapi_cleaning file and key features were isolated and explored in the nyt_eda files. Such key features included: keywords, bylines, leading paragraphs, word count, news desk, headline, etc. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The baseline model assigned front page or rest of paper based on the known statistical distribution (~6% front page). It was determined to have an accuracy of 89.56%. Because the accuracy of the baseline model was so high and the data was heavily balanced in favor of articles not on the front page, other metrics, such as precision, recall, and F1 score were used as better measures of improvement over baseline. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Many modeling options were used to considered to create a model which improved the most over the baseline model: bag of words, random forest, logistic regression, neural network, etc. After experimentation, logistic regression, recurrent neural network, and XGBoost models were decided upon and further developed. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;All models showed advantages over the baseline model in some way. All models, however, did show some degree of overfitting, with XGBoost showing the least overfitting (<5%). After removing word_count from the models which was considered a highly biased feature, the logistic regression model with the TfidfVectorizer fine tuned. It resulted in a recall of 70.27%, precision of 77.97%, and an F1-score of 73.71.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This model is very much still in the proof-of-concept stage and has not been developed so that users could input the required features and receive a prediction of whether the article in question would be on the front page. Further work would be needed to get the model to this point.

**Key Results:**

This project did show that a model could be created to predict whether an article will be printed on the front page of the New York Times. Over reliance on word_count could bias the model, causing overfitting. It was determined that articles that mention American politics, cities, and goverment are likely to be deemed "important" and be placed on the front page. 

**Future Work:**

Should this project continue, there are many avenues that could be explored. The group could look at understanding how the time dimension plays a role in missclassification. More effective word processing could be pursued by looking at how the removal of filler words impacts the model. Vector autoregressive article representation can be looked at to further explore semantic meaning within text features. Lastly, the group could look at interpretability and developing a method for the model to be used by people outside the development group. 

**Repo Table of Contents:**

Main Branch
1)	baseline_logistic_regression.ipynb - baseline model and logistic regression model
2)	Final Presentation.pdf - results as of 07Aug2024 presented
3)	GBmodels.py - balancing data, pre-processing
4)	GBmodels2.py - model optimization with lime
5)	GBmodels3 - XGBoost model using text embeddings
6)	GBmodels4 - XGBoost model using frequency extraction on text
7)	RNNmodels.ipynb - RNN model diagnostics
8)	RNNmodels2.ipynb - RNN model

EDA Folder
1)	nytapi_cleaning.ipynb – API web scraping and initial data cleaning
2)	nyt_eda.ipynb – key word extraction
3)	nyt_eda_2_byline_cleaning.ipynb – byline extraction
4)	nyt_eda_3_wordcount_leadpara.ipynb – word count statistics and time dependency work on lead paragraphs
5)	nyt_eda_general_statistics.ipynb – percentages of N/A values on front page vs rest of paper, character counts for lead paragraph/snippet, unique values determined
6)	save_data.py - preprocesses, filters, and creates derived features from the raw data file available for download


**Credits:**
Alec Naidoo (@Alec12), Philip Monaco (@Philip-Monaco), Sarah Julius(@sarahcj94), and Shruti Gupta (@sguptaray) 

**Appendix A – API Feature Documentation:**

Web_url: string, article url

Abstract: string, summary of article

Snippet: string, 0-250 characters, commonly the same as or a portion of the abstract

Lead_paragraph: first paragraph of the article, string, averages ~300 characters, often starts with location

Print_section: string, section of the newspaper the article is in\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	A: Front Page, International News, National News, Obituaries, Editorials, Op-Eds, and Letters, Corrections\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	B: Business Day, Sports, Obituaries\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	C: Arts, Weekend Arts\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	D: Science Times, Sports, Style, Food\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• MB: New York (Sunday Times)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	AR: Arts and Leisure (Sunday Times)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	BU: Business (Sunday Times)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	SR: Opinion (Sunday Times)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	MM: Magazine (Sunday Times)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	BR: Book Review (Sunday Times)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	SP: Sports (Sunday Times)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	ST: Style, Vows (Sunday Times)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	RE: Real Estate (Sunday Times)

Print_page: page within section article is located

Source: string, where the article was acquired from all instances are The New York Times in this dataset

Headline: dictionary of strings\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	Main\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	Kicker: often none\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	Content_kicker: often none\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	Print_headline: often the same as main\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	Name: often none\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	Seo: often none\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	Sub: often none

Keywords: dictionary of dictionaries\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	Name: common values include person, glocations, subject, organizations\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	Value\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	Rank\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	Major

Newsdesk/Section_name: similar values, strings that represent what article is about; National, Foreign, Sports, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Metro, Business, Sports, Letters, Dining, Society, etc

Subsection_name: string, optional, further classifies what article is about

Byline: dictionary of strings and dictionaries\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	Original: byline as written\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	Person: extracted first, middle, last names, etc from original\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•	Organization: often none or the New York Times

Type_of_material: string, News, Question, Letter, Biography, List, Summary, Review, etc

**Appendix B – Created Feature Documentation:**

year: extracted from pub_date, used to determine training, validation, and testing data

combined_text: headline, snippet, and lead parahraph concatenated together

num_subj: count of subjects in keyword, created from information extracted from keywords

num_persons: count of people mentioned in keywords, created from information extracted from keywords

num_glocs: count of locations in keywords, created from information extracted from keywords

byline_extract: extracted from the original byline entry in the byline dictionary

byline_confirm: extracted from either person or organization entry in  in the byline dictionary

byline_match: used to determine if there were rows where information in the original byline and the &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;person/organization do not line up

headline_len: determined from headline entry

lead_paragraph_length: determined from lead_paragraph entry

pub_hour/pub_day/pub_day_of_month/pub_month/pub_year: determined from pub_date using datetime functions

text_features: list of title and lead_paragraph

categorical_features: list of type_of_materials and author

numeric_features: list of headline_length, lead_paragraph_length, pub_hour, pub_day, pub_day_of_month, pub_month, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pub_year, pub_day
