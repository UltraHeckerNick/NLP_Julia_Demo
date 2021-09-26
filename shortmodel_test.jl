#Title: Natural Language Processing for Language Detection
#Imports: Nicholas Trejo, language,phrase-dataset. Based on Wiktionary data: https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists
#Author: Nicholas Trejo, August 2021

# OPERATING INSTRUCTIONS!!! To start file, ensure training data CSV file is in directory of the Julia source. 
# Recommend changing directory in terminal prior to invoking Julia REPL; invoke using [julia]. 

# Prior to [include] function call, enter the following to ensure packages are in current path:
# Pkg.add("TextAnalysis") 
# Pkg.add("CSV") 
# Pkg.add("DataFrames") 

# Once in correct directory with packages installed, and Julia REPL invoked, use [include("custom_model_test.jl")].
# In Julia REPL, type [langdetect()] to call function.
# Once prompted, enter test phrase and the results of Naive-Bayes prediction is returned.


# Tells the REPL what packages will be used.
using CSV, DataFrames
using TextAnalysis: NaiveBayesClassifier, fit!, predict

# Declaration of Model; Naive-Bayes modeling is used [https://en.wikipedia.org/wiki/Naive_Bayes_classifier]
language_data_unprocessed = CSV.File("word.csv") |>DataFrame
language_data = remove_punctuation!(language_data_unprocessed)
global model = NaiveBayesClassifier([:english, :filipino, :french, :german, :haitian, :portuguese, :spanish, :vietnamese])

# relates test phrase with specific class to build classification model.
for row in eachrow(language_data)
    if row.language == "english"
        fit!(model, filter(isvalid, row.phrase), :english)
    elseif row.language == "filipino"
        fit!(model, filter(isvalid, row.phrase), :filipino)
    elseif row.language == "french"
        fit!(model, filter(isvalid, row.phrase), :french)
    elseif row.language == "german"
        fit!(model, filter(isvalid, row.phrase), :german)
    elseif row.language == "haitian"
        fit!(model, filter(isvalid, row.phrase), :haitian)
    elseif row.language == "portuguese"
        fit!(model, filter(isvalid, row.phrase), :portuguese)
    elseif row.language == "spanish"
        fit!(model, filter(isvalid, row.phrase), :spanish)
	elseif row.language == "vietnamese"
        fit!(model, filter(isvalid, row.phrase), :vietnamese)
    end
end

# Prompts user to enter test phrase, then displaying prediction results. 
# Prompt input is automatically read as type::string.
function langdetect()
	while true
		println("Please enter your test phrase:")
		test_phrase = readline()
		prediction = predict(model,test_phrase)
		sort_pred = reverse(sort(collect(prediction), by=x->x[2]))
		println("Top three language prediction:")
		println(sort_pred[1])
		println(sort_pred[2])
		println(sort_pred[3])
	end
end
