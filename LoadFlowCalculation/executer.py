
import subprocess

try:
    while True:
        # Starte das Python-Programm und warte auf dessen Beendigung
        subprocess.run(['python', 'main_powerfactory.py'])
except KeyboardInterrupt:
    print("\nProgram was stopped by user input.")