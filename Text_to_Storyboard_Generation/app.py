from flask import Flask, render_template, request, jsonify
from models.text_module import TextUnderstandingModule
from models.image_module import ImageGenerationModule
from utils.preprocess import preprocess_text

app = Flask(__name__)

# Load trained models
text_module = TextUnderstandingModule(input_size, hidden_size, num_layers, num_classes)
text_module.load_state_dict(torch.load('saved_models/text_module.pth'))
text_module.eval()

image_module = ImageGenerationModule(latent_dim, img_channels, img_size)
image_module.load_state_dict(torch.load('saved_models/image_module.pth'))
image_module.eval()

# Define route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Define API endpoint for generating storyboards
@app.route('/generate_storyboard', methods=['POST'])
def generate_storyboard():
    data = request.get_json()
    input_text = data['text']

    # Preprocess input text
    text_inputs = preprocess_text(input_text)

    # Generate storyboard
    with torch.no_grad():
        text_features = text_module(text_inputs)
        generated_storyboard = image_module(text_features)

    # Convert generated storyboard to base64 string or other suitable format for transmission
    # Example: generated_storyboard_base64 = convert_to_base64(generated_storyboard)

    return jsonify({'generated_storyboard': generated_storyboard})

if __name__ == '__main__':
    app.run(debug=True)
