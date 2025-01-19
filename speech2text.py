# Python program to translate
# speech to text and text to speech


import speech_recognition as sr
import pyttsx3 

# Initialize the recognizer 
r = sr.Recognizer() 

# Function to convert text to
# speech
def speak(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    engine.say(command) 
    engine.runAndWait()
    
def listen():
    try:
        # use the microphone as source for input.
        with sr.Microphone() as source2:
                
            # wait for a second to let the recognizer
            # adjust the energy threshold based on
            # the surrounding noise level 
            r.adjust_for_ambient_noise(source2, duration=0.5)
            
            #listens for the user's input 
            audio2 = r.listen(source2, 60, 30)
            
            # Using google to recognize audio
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()

            # print("Did you say ",MyText)
            # speak(MyText)
            return MyText
            
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        
    except sr.UnknownValueError:
        print("unknown error occurred")

if __name__ == "__main__":
    listen()