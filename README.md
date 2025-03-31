# Cloud Detection App

A web application that uses computer vision to detect and analyze clouds in images. Built with Flask and OpenCV.

## Features

- Upload and process images
- Cloud detection and analysis
- Mobile-friendly interface
- Progressive Web App (PWA) support
- Cross-platform compatibility

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cloud-detection-app.git
cd cloud-detection-app
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python src/app.py
```

The app will be available at `http://localhost:5000`

## Usage

1. Open the web application in your browser
2. Click "Choose File" to select an image
3. Click "Upload" to process the image
4. View the cloud detection results and analysis

## Mobile Installation

### Android
1. Open the web app in Chrome
2. Click the menu (three dots)
3. Select "Add to Home screen"
4. Follow the prompts to install

### iOS
1. Open the web app in Safari
2. Click the share button
3. Select "Add to Home Screen"
4. Follow the prompts to install

## Development

The project structure is organized as follows:
```
cloud-detection-app/
├── src/
│   └── app.py
├── static/
│   ├── css/
│   ├── js/
│   └── manifest.webmanifest
├── templates/
│   └── index.html
├── requirements.txt
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 