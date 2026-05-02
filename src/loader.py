from langchain_community.document_loaders import PyPDFLoader

def load_data(file_path):
    if not file_path.endswith('.pdf'):
        raise ValueError("Unsupported file format. Please provide a PDF file.")
    else:
        print(f"Loading data from {file_path}...")
        pdf = PyPDFLoader(file_path)
        document = pdf.load()
        print('Successfully loaded document with', len(document), 'pages.')
        return document
    

# Example usage:
# load_data('YOUR_FILE.pdf')