#import 
import tkinter as tk
import tkinter.ttk as ttk
import pandas as pd
from tkinter.filedialog import askopenfile


file_name = None

def set_file_name(file_path):
    global file_name
    file_name = file_path.name

def get_file_name():
    return file_name

class Dashboard(tk.Frame):
    def __init__(self, master =None):
        super().__init__(master)
        self.configure(background="white")
        def import_csv_data():
            file_path = askopenfile(mode = 'r', filetypes = [('XLSX Files','*xlsx' ),('CSV Files', '*.csv'), ("All Files",'*') ], title = "Choose a file")
            set_file_name(file_path)

            if file_path:
                MainScreen(self.master)

         # Title
        title = tk.Label(self, text="Naive Bayes Classifier Prediction", font=("Arial", 20), bg="white")
        title.grid(row=0, column=0, columnspan=2, pady=20)

        # Button
        btn1 = tk.Button(self, text="Import Dataset", font=("Arial", 15), bg="blue", fg='#ffffff', command=import_csv_data)
        btn1.grid(row=1, column=1, padx=150, pady=20)
        self.pack()

       
class MainScreen(tk.Toplevel):
    fileName = None

    
    def __init__(self, master = None):
        super().__init__(master)
        self.master.withdraw()
        self.geometry("1300x600")
        
        def disable_event():
            pass    
        
        self.protocol("WM_DELETE_WINDOW", disable_event)
        file = tk.Label(self, text="File Name: ", font=("Arial", 10))
        file.grid(row=0, column=0, padx=30, pady=10)

        # Disabled entry for filename
        entry_file = tk.Entry(self, width=100)
        entry_file.insert(0, get_file_name())
        entry_file.configure(state="disabled")
        entry_file.grid(row=0, column=1, pady=10, columnspan=2)

        # Text area for data
        head_table = tk.Text(self, width=70, height=15 , yscrollcommand=True, xscrollcommand=True, spacing2=5)
        head_table.grid(row=1, column=1, columnspan=2, padx=20, pady=10)
        # Text area for classification report
        classification_report_table = tk.Text(self, width=60, height=15 , yscrollcommand=True, xscrollcommand=True)    
        classification_report_table.grid(row=1, column=3, columnspan=2, pady=10)
        

        # check if xls or csv
        file_name = get_file_name()
        if file_name.endswith('.csv'):
            df = pd.read_csv(get_file_name())
            head_table.insert(tk.END, df.head(10))

        elif file_name.endswith('.xlsx'):
            df = pd.read_excel(get_file_name())
            head_table.insert(tk.END, df.head(10))

        # Label size data
        size = tk.Label(self, text="Size Data:", font=("Arial", 10))
        size.grid(row=2, column=0, padx = 5, pady=10)

        # Ukuran data
        size_data = tk.Entry(self, width=40)
        size_data.insert(0, df.size)
        size_data.configure(state="disabled")
        size_data.grid(row=2, column=1, pady=10)

        # Label variabel independen
        label_x = tk.Label(self, text="Variabel Independen", font=("Arial", 10))
        label_x.grid(row=3, column=0, padx = 20,pady=10)

        # variabel independen
        entry_x = tk.Entry(self, width=40)
        data_independen = ""
        for i in df.columns[:-1]:
            data_independen += i + ", "
        entry_x.insert(0, data_independen)
        entry_x.configure(state="disabled")
        entry_x.grid(row=3, column=1, padx=20, pady=10)
        
        # Label variabel dependen
        label_y = tk.Label(self, text="Variabel Dependen", font=("Arial", 10))
        label_y.grid(row=4, column=0, padx = 20,pady=10)

        # variabel dependen
        entry_y = tk.Entry(self, width=40)
        entry_y.insert(0, df.columns[len(df.columns)-1])
        entry_y.configure(state="disabled")
        entry_y.grid(row=4, column=1, padx=20, pady=10)


        # Label set_test_size
        set_test_size = tk.Label(self, text="Set Test Size:", font=("Arial", 10))
        set_test_size.grid(row=3, column=2, padx = 20,pady=10)    


        # Scaler test_size
        test_size_sc = tk.Scale( self,from_=0.01, to=1.00, digits = 3, resolution = 0.01,
           orient = tk.HORIZONTAL, length = 180, width = 15, sliderlength = 20) 
        test_size_sc.grid(row=3, column=3, padx=20, pady=10)
        test_size_sc.set(0.05)

        # Label random_state
        label_random_state = tk.Label(self, text="Random State:", font=("Arial", 10))
        label_random_state.grid(row=4, column=2, padx = 20,pady=10)  

        # random_state
        random_state = tk.Entry(self, width=30)
        random_state.insert(0, 42)
        random_state.grid(row=4, column=3, padx=20, pady=10)

        # Button classification report
        btn_classification_report = tk.Button(self, text="Classification Report", font=("Arial", 10), bg="blue", fg='#ffffff', command=lambda: self.classification_report(test_size_sc.get(), random_state.get(), classification_report_table))
        btn_classification_report.grid(row=2, column=3, padx=20, pady=10)

        # Button Confusion Matrix
        btn_confusion_matrix = tk.Button(self, text="Confusion Matrix", font=("Arial", 10), bg="blue", fg='#ffffff', command=lambda: self.confusion_matrix(test_size_sc.get(), random_state.get()))
        btn_confusion_matrix.grid(row=5, column=3, padx=20, pady=10)

        # Button Accuracy
        btn_accuracy = tk.Button(self, text="Accuracy", font=("Arial", 10), bg="blue", fg='#ffffff', width = 15, command=lambda: self.accuracy(test_size_sc.get(), random_state.get()))
        btn_accuracy.grid(row=5, column=2, padx=20, pady=10)

        # Button Close
        btn_close = tk.Button(self, text="Close", font=("Arial", 10), bg="red", fg='#ffffff', width = 15, command=lambda: self.master.destroy())
        btn_close.grid(row=5, column=1, padx=20, pady=10)

        self.mainloop()


    # Menghitung nilai akurasi
    def accuracy(self,test_size_sc, random_state):
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score

        file_name = get_file_name()
        if file_name.endswith('.csv'):
            df = pd.read_csv(get_file_name())
        elif file_name.endswith('.xlsx'):
            df = pd.read_excel(get_file_name())

        df = df.dropna()


        # make sure the data type are integer
        rs = int(random_state)
        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=test_size_sc, random_state=rs)

        #  Model
        model = GaussianNB().fit(X_train, y_train)

        # Predicted_y
        predicted_y = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, predicted_y)

        # notif
        notif = tk.Toplevel()
        notif.geometry("300x100")
        notif.resizable(False, False)
        notif.configure(background="white")
        notif.title("Accuracy")
        
        label = tk.Label(notif, text="Accuracy: " + str(accuracy), font=("Arial", 10))
        label.grid(row=0, column=0, padx=20, pady=10)
        
        btn = tk.Button(notif, text="OK", font=("Arial", 10), bg="blue", fg='#ffffff', width =10, command=notif.destroy)
        btn.grid(row=1, column=0, padx=20, pady=10)


    # Menghitung nilai classification report
    def classification_report(self, test_size, random_state, table):
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import classification_report

        file_name = get_file_name()
        if file_name.endswith('.csv'):
            df = pd.read_csv(get_file_name())
        elif file_name.endswith('.xlsx'):
            df = pd.read_excel(get_file_name())

        df = df.dropna()
        
        rs = int(random_state)
        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=test_size, random_state=rs)

        #  Model
        model = GaussianNB().fit(X_train, y_train)

        # Predicted_y
        predicted_y = model.predict(X_test)

        # Classification Report
        classification_report = classification_report(y_test, predicted_y)
        # if there is a table, clear the data
        if table.get("1.0", tk.END) != "":
            table.delete("1.0", tk.END)
        #  fill the table
        table.insert(tk.END, classification_report)

        
    # Menghitung confusion matrix
    def confusion_matrix(self, test_size, random_state):
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import confusion_matrix

        file_name = get_file_name()
        if file_name.endswith('.csv'):
            df = pd.read_csv(get_file_name())
        elif file_name.endswith('.xlsx'):
            df = pd.read_excel(get_file_name())

        # clean the data
        df = df.dropna()

        rs = int(random_state)
        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=test_size, random_state=rs)

        #  Model
        model = GaussianNB().fit(X_train, y_train)

        # Predicted_y
        predicted_y = model.predict(X_test)

        # Confusion Matrix
        confusion_matrix = confusion_matrix(y_test, predicted_y)

        # notif
        notif = tk.Toplevel()
        notif.geometry("300x100")
        notif.resizable(False, False)
        notif.configure(background="white")
        notif.title("Confusion Matrix")
        
        label = tk.Label(notif, text="Confusion Matrix: " + str(confusion_matrix), font=("Arial", 10))
        label.grid(row=0, column=0, padx=20, pady=10)
        
        btn = tk.Button(notif, text="OK", font=("Arial", 10), bg="blue", fg='#ffffff',width =10, command=notif.destroy)
        btn.grid(row=1, column=0, padx=20, pady=10)

if __name__ == '__main__':
    main_window = tk.Tk()
    main_window.title("Naive Bayes Classifier Prediction")
    main_window.geometry("500x200")
    main_window.resizable(False, False)
    main_window.configure(background="white")

    dashboard = Dashboard(main_window)
    dashboard.mainloop()