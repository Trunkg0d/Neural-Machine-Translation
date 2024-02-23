# Neural Machine Translation

## Description
This repository contains the implementation of a Neural Machine Translation (NMT) system for translating text from English to Portuguese, employing the sequence-to-sequence (Seq2Seq) model with attention mechanism. Neural Machine Translation has seen significant advancements in recent years, with Seq2Seq models becoming the cornerstone due to their ability to effectively translate sequences of variable length.

The Seq2Seq model, coupled with attention, has revolutionized the field of machine translation by allowing the model to focus on relevant parts of the input sentence while generating the corresponding translation. This attention mechanism enhances the model's ability to capture long-range dependencies and improves the quality of translations, particularly for long sentences or sentences with complex structures.

## Getting Started

### Dependencies

To orchestrate this symphony of transformation, ensure you have the required packages installed. Create a new Anaconda environment and install the packages using the following command:
* Create new env and install pip:
```
conda create --name <your_env_name> pip
```
* Activate your env:
```
conda activate <your_env_name>
```
* Install packages
```
pip install -r requirements.txt
```

### Installing

Download the project from GitHub either as a zip folder or by executing the following command:
```
git clone https://github.com/Trunkg0d/Neural-Machine-Translation.git
```

### Executing program
Before using the translation system, we need to train our model and save its weights. Navigate to the src directory and execute the following command:
```
cd .\src\
python train.py
```
Once the training is complete and the weights are saved, you can launch the web user interface (UI) by executing:
```
streamlit run main.py
```
This will start the UI, allowing you to input English text and receive its corresponding translation in Portuguese. Enjoy exploring the wonders of neural machine translation! 🚀

## Authors

Contributors names and contact info

[@Trunkg0d](https://www.facebook.com/htak2003)

## License

## Acknowledgments
Eternal gratitude to the wizards and sorceresses of open-source magic.
