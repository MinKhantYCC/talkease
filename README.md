# Talk Ease

AI chatbot powered by a Large Language Model (TinyLlama-1.1B-chat).
It is used to create an active English environment for English learners who are seeking a speaking partner to practice their speaking and listening skill.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Features
- Audio input
- Audio output
- Speech-to-text

## Installation
1. Clone the repository:
```bash

git clone git@github.com:MinKhantYCC/talkease.git
cd talkease
```
2. Install dependencies:
```bash

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
To start hosting the streamlit application, run:
```bash

streamlit run main.py
```
Press the `Start recording` button to start talking.
Press the `Stop recording` button after you finish talking.
The application will response with an audio.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m 'Add feature-name'`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
Special thanks to:
- [TheBloke](https://huggingface.co/TheBloke)
- [Leon Explains AI](https://youtu.be/CUjO8b6_ZuM)

## Contact
For questions or support, contact: minkhant.nexusjoint@gmail.com