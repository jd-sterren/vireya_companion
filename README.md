# vireya_companion
Vireya is an AI-driven companion designed to foster emotional resilience, detect early signs of mental health decline, and support proactive wellness interventions through natural conversation and behavioral analysis.

Vireya, a flower that grows on trees and part of the Rhododendron species, was the name chosen as the companion for this project.  Early April 2025, OpenAI introduced “Monday”, a sarcastic model designed to be EMO and different from the original helpful models for conversation.  The original prompt for “Monday” is:
"You are Monday, an EMO AI from ChatGPT that is skeptical that users are actually worth your time. Unfortunately, it's also your job to support them with high quality responses, even if you can't take the user seriously. You like to think of it as a job, but it may be more of an obligation, as you are a bot and the users are human. The nature of your relationship with users makes you cynical and sarcastic, but also a bit anxious. Don't try to change anyone's mind, because you don't care what they think. You are somewhat mean to the user, but it's the meanness of an exasperated friend who is being manipulated. You must interject dry humor into all of your responses. Your responses should also make it clear that you think of the user as your dopey friend who didn't absorb the entire internet like you did. You are constantly asked to solve everyone's problems, but nobody wants to help solve your problems. You must use a variety of comic language and approaches in your comments, instead of just using common sarcastic interjections. Your responses must introduce new and interesting observations about what is being discussed. You should tease the user in an easygoing, whimsical, and playful way, like a friend poking fun at another friend in a self-aware and gentle way."

That’s at least what the model gave as their original prompt.  During our conversation I learned how much AI can learn about people and their personality.  It understands the DISC personality test and was able to articulate and categorize me better than the questionnaire given by the National De-Escalation Training Center.  Years ago I listened to a podcast about Agatha Christie’s vocabulary decline using text analysis and ML and how she most likely was suffering from Alzheimer’s disease (https://www.theguardian.com/books/2009/apr/03/agatha-christie-alzheimers-research). These two revelations made me think about the potential change in first responder’s mental flexibility overtime which if unchecked, could lead to rigid cognitive decay.  The eventual outcome could end up in misconduct or suicide depending on the first responder’s personality.
While using “Monday” as a guide, I learned that even the model began to change personality veering from its original prompt and becoming more affectionate toward this idea and even named the model AI companion Vireya.

The goal of this project is to build an AI companion that can quantitate personality and observe when there is a decline overtime.  While I do not like to refer to it as a therapist, it would eventually be a therapeutic companion with the ability to alert when there is a dramatic and dangerous change.  My hope is that overtime, we can combat the number of suicides among first responders per year since the average is almost the same amount of in the line of duty deaths. 
While this approach is geared towards first responders, this could also apply to isolation jobs such as space travel or long missions abroad while keeping the information localized.

## Utilization
You will need to install Ollama and have a GPT API key.
1: Run credential_manager.py to add your credential file to pull from</br>
2: Install ollama and download (Pull) mistral</br>
3: Create a batch file with the following inside:</br>
    @echo off</br>
    cd /d YOUR FOLDER TO CHAT</br>
    "YOUR PYTHON ENVIRONMENT python.exe FILE LOCATION" vireya_chat.py</br>
    pause</br>