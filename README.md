# Sentimental-Chatbot
Project to create a chatbot that generates responses based on the sentiment gathered from the users' input


Setup:
- Access files from following: https://drive.google.com/drive/u/2/my-drive
- If on hyperion, run <sbatch create_env.sh> with Sentimental-Classifier in the username folder (aaaabbb)
otherwise do pip3 install -r requirements.txt in the environments

Inference:
1) Place files based on the corresponding folders (vocabulary_senti_classifier.pkl > /Sentimental-Chatbot/) etc.
2) test.py are available for Chatbot and Sentimental Classifier folders, both are baseline models
    - Available configurations for chatbot with established checkpoints for chatbot:
    - n_iterations: 80000
    - lstm : [True, False]
    - use_attention : [True, False]
    - method : [0, 1, 2] for lstm = False only

    - Avaialble configurations for sentimental classifier with established checkpoints:
    - regression : [True,False]

3) Run on training loop, input "q" or "quit" to exit, or just stop the code entirely

3) inference.py is the final test file for the final model in the Sentimental Chatbot
    - Avaiable checkpoints:
    - lstm = True, use_attention = True, method = 0, n_iteration = 10000
    - lstm = False, use_attention = True, method = 0, n_iteration = [20000, 90000, 100000] (Better performance)

4) run_job_Chatbot for train.py under Chatbot/, run_job_Senti for train.py under Senti_Classifier/, run_job_SentiChatbot for train_SC, files are run on Hyperion under the username folder (aaaabbb), with Sentimental Chatbot also in the username folder
