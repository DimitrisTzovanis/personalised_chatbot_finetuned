# Personalised Chatbot, finetuned from GPT-2 model

## Members: Dimitris Tzovanis

#### Create your own chtbot, on your own conversation data (both in english and greek), by finetuning a big language model 




#### Finetuning the pretrained model

There are 2 python files in this project, specifically for creating the models. One for finetuning a model pretrained in greek, and one in english (DialoGPT model). 
In order to run the scripts, edit the script and put the name of your train and validation json files. 
To help with this step, there is a python script included that takes as input a csv file and outputs the 2 json files as needed.
The result is a 500MB folder, containing the output model.

To run the model, execute the relative use_x_model python script
