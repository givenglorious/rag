from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader,CSVLoader,TextLoader,Docx2txtLoader,JSONLoader

class loader_data():
    def __init__(self, file_path):
        self.file_path = file_path
        print(f"Initialized loader for file: {file_path}")
        
    def load_data(file_path: str):
        ext = file_path.split(".")[-1].lower()
        loader = loader_data(file_path)

        dispatch = {
            "pdf":  loader.load_pdf,
            "csv":  loader.load_csv,
            "txt":  loader.load_txt,
            "docx": loader.load_docx,
            "json": loader.load_json,
        }

        if ext not in dispatch:
            raise ValueError(f"Unsupported file type: .{ext}")

        return dispatch[ext]()
    
    def load_pdf(self):
        if not self.file_path.endswith('.pdf'):
            raise ValueError("Unsupported file format. Please provide a PDF file.")
        else:
            print(f"Loading data from {self.file_path}...")
        pdf = PyPDFLoader(self.file_path)
        document = pdf.load()
        print('Successfully loaded document with', len(document), 'pages.')
        return document
    
    def load_csv(self):
        if not self.file_path.endswith('.csv'):
            raise ValueError("Unsupported file format. Please provide a CSV file.")
        else:
            print(f"Loading data from {self.file_path}...")
        csv = CSVLoader(file_path=self.file_path)
        document = csv.load()
        print('Successfully loaded document with', len(document), 'pages.')
        return document
        
    def load_txt(self):
        if not self.file_path.endswith('.txt'):
            raise ValueError("Unsupported file format. Please provide a text file.")
        else:
            print(f"Loading data from {self.file_path}...")
        txt = TextLoader(file_path=self.file_path)
        document = txt.load()
        print('Successfully loaded document with', len(document), 'pages.')
        return document
    def load_docx(self):
        if not self.file_path.endswith('.docx'):
            raise ValueError("Unsupported file format. Please provide a DOCX file.")
        else:
            print(f"Loading data from {self.file_path}...")
        docx = Docx2txtLoader(file_path=self.file_path)
        document = docx.load()
        print('Successfully loaded document with', len(document), 'pages.')
        return document

    def load_json(self):
        if not self.file_path.endswith('.json'):
            raise ValueError("Unsupported file format. Please provide a JSON file.")
        else:
            print(f"Loading data from {self.file_path}...")
        json = JSONLoader(file_path=self.file_path,jq_schema=".")
        document = json.load()
        print('Successfully loaded document with', len(document), 'pages.')
        return document
    

# Example usage:
# loader = loader_data('YOUR_FILE.pdf')
# loader.load_data()