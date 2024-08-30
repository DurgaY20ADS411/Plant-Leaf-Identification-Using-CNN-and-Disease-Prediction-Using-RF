document.getElementById('uploadForm').addEventListener('submit', function(event) {
            event.preventDefault();
            var fileInput = document.getElementById('fileInput');
            var file = fileInput.files[0];
            var formData = new FormData();
            formData.append('file', file);
			
			
			// Check if a file is uploaded
            if (!file) {
                alert('Please select an image file.');
                return;
            }

            // Check if the uploaded file is not a leaf image
            var fileExtension = file.name.split('.').pop().toLowerCase();
            if (fileExtension !== 'jpg' && fileExtension !== 'jpeg' ) {
                alert('Please upload a valid image file (JPEG or JPG).');
                return;
            }
			

            fetch('/predict_cnn', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
			
			
                var outputDiv = document.getElementById('output');
                outputDiv.innerHTML = '';
                var img = document.createElement('img');
                img.src = URL.createObjectURL(file);
                outputDiv.appendChild(img);
				
                
				
				
				var p = document.createElement('p');
                p.textContent = 'Predicted Disease: ' + data.predicted_class;
                outputDiv.appendChild(p);
				
				
            })
            .catch(error => console.error('Error:', error));
        });