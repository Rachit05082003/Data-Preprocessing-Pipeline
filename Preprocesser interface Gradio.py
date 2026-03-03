import pandas as pd
import gradio as gr
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from dateutil.parser import parse
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    def __init__(self, data_source):
        self.data_source = data_source
        self.data = None
        self.label_encoders = {}
        self.selected_columns = {
            "numeric": [],
            "categorical": []
        }
        self.columns_to_process = {
            "numeric": [],
            "categorical": [],
            "date": [],
            "excluded": []
        }

    def load_data(self):
        if isinstance(self.data_source, pd.DataFrame):
            self.data = self.data_source.copy()
            print("Data loaded successfully.")
        else:
            raise TypeError("Invalid data source. Provide a DataFrame.")

    def identify_columns(self):
        excluded_keywords = ['id', 'number', 'code', 'identifier']

        for column in self.data.columns:
            col_dtype = self.data[column].dtype
            col_name = column.strip().lower()

            if col_dtype in ['int64', 'float64']:
                if any(keyword in col_name for keyword in excluded_keywords):
                    self.columns_to_process["excluded"].append(column)
                else:
                    self.columns_to_process["numeric"].append(column)
            elif col_dtype == 'object':
                if self.is_date_column(column):
                    self.columns_to_process["date"].append(column)
                else:
                    if any(keyword in col_name for keyword in excluded_keywords):
                        self.columns_to_process["excluded"].append(column)
                    else:
                        self.columns_to_process["categorical"].append(column)
            else:
                self.columns_to_process["excluded"].append(column)

        print(f"Numeric Columns for Scaling: {self.columns_to_process['numeric']}")
        print(f"Categorical Columns for Encoding: {self.columns_to_process['categorical']}")

    def is_date_column(self, column):
        try:
            sample = self.data[column].dropna().head(10).astype(str)
            parsed_count = sum(1 for val in sample if self.is_valid_date(val))
            return parsed_count / len(sample) >= 0.8
        except Exception:
            return False

    def is_valid_date(self, value):
        try:
            parse(value, fuzzy=False)
            return True
        except Exception:
            return False

    def scale_numeric_data(self, selected_columns):
        if selected_columns:
            scaler = StandardScaler()
            self.data[selected_columns] = scaler.fit_transform(self.data[selected_columns])
            print(f"Scaled columns: {selected_columns}")
            return f"Scaled columns: {selected_columns}"
        else:
            return "No columns selected for scaling."

    def label_encode_categorical_data(self, selected_columns):
        if selected_columns:
            for column in selected_columns:
                # Ensure all values are strings to avoid type errors
                self.data[column] = self.data[column].astype(str)
                encoder = LabelEncoder()
                self.data[column] = encoder.fit_transform(self.data[column])
                self.label_encoders[column] = encoder
                print(f"Encoded column: {column}")
            return f"Encoded columns: {selected_columns}"
        else:
            return "No columns selected for encoding."
        #Written by Rachit
    def format_date_columns(self, date_format='%d/%m/%Y'):
        if self.columns_to_process["date"]:
            with ThreadPoolExecutor() as executor:
                def format_column(column):
                    def normalize_date(value):
                        try:
                            value = str(value).replace('st', '').replace('nd', '').replace('rd', '').replace('th', '')
                            return parse(value, fuzzy=True)
                        except Exception:
                            return None

                    self.data[column] = self.data[column].apply(normalize_date)
                    self.data[column] = pd.to_datetime(self.data[column], errors='coerce')
                    self.data[column] = self.data[column].dt.strftime(date_format)
                executor.map(format_column, self.columns_to_process["date"])
            print(f"Formatted date columns to format {date_format}.")

    def drop_missing_rows(self):
        initial_row_count = len(self.data)
        self.data.dropna(inplace=True)
        final_row_count = len(self.data)
        print(f"Dropped {initial_row_count - final_row_count} rows with missing values.")
        return f"Dropped {initial_row_count - final_row_count} rows with missing values."

    def drop_empty_columns(self):
        initial_column_count = self.data.shape[1]
        self.data.dropna(axis=1, how='all', inplace=True)
        final_column_count = self.data.shape[1]
        print(f"Dropped {initial_column_count - final_column_count} empty columns.")
        return f"Dropped {initial_column_count - final_column_count} empty columns."

    def fill_missing_values(self, numeric_strategy='mean', categorical_strategy='mode', custom_values=None):
        if custom_values is None:
            custom_values = {}

        with ThreadPoolExecutor() as executor:
            def fill_numeric(column):
                if column in custom_values:
                    self.data[column].fillna(custom_values[column], inplace=True)
                elif numeric_strategy == 'mean':
                    self.data[column].fillna(self.data[column].mean(), inplace=True)
                elif numeric_strategy == 'median':
                    self.data[column].fillna(self.data[column].median(), inplace=True)

            def fill_categorical(column):
                if column in custom_values:
                    self.data[column].fillna(custom_values[column], inplace=True)
                elif categorical_strategy == 'mode':
                    self.data[column].fillna(self.data[column].mode()[0], inplace=True)

            executor.map(fill_numeric, self.columns_to_process["numeric"])
            executor.map(fill_categorical, self.columns_to_process["categorical"])

        print("Filled missing values in the dataset.")
        return "Filled missing values in the dataset."

    def perform_pca(self):
        if not self.columns_to_process["numeric"]:
            return "No numeric columns available for PCA.", None, None, None

        try:
            # Step 1: Standardize the data
            scaler = StandardScaler()
            numeric_data = self.data[self.columns_to_process["numeric"]]
            scaled_data = scaler.fit_transform(numeric_data)

            # Step 2: Perform PCA
            n_components = scaled_data.shape[1]  # Use all features as components
            self.pca = PCA(n_components=n_components)  # Save the PCA model
            self.scaler = scaler  # Save the scaler
            self.scaled_data = scaled_data  # Save the scaled data
            pca_result = self.pca.fit_transform(scaled_data)
            self.pca_result = pca_result  # Save PCA components

            # Step 3: Calculate cumulative variance
            cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)

            # Plot cumulative variance
            plt.figure(figsize=(10, 6))
            plt.plot(
                range(1, len(cumulative_variance) + 1),
                cumulative_variance,
                marker="o",
                linestyle="--"
            )
            plt.title("Cumulative Variance Explained by PCA Components")
            plt.xlabel("Number of Principal Components")
            plt.ylabel("Cumulative Variance Explained")
            plt.grid()
            cumulative_variance_plot_path = "cumulative_variance_plot.png"
            plt.savefig(cumulative_variance_plot_path)
            plt.close()

            # Step 4: Visualize PCA loadings
            loadings = pd.DataFrame(
                self.pca.components_,
                columns=self.columns_to_process["numeric"],
                index=[f"PC{i + 1}" for i in range(n_components)]
            )
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                loadings,
                annot=True,
                cmap="coolwarm",
                cbar=True,
                fmt=".2f",
                linewidths=0.5
            )
            plt.title("PCA Components and Feature Loadings")
            plt.xlabel("Features")
            plt.ylabel("Principal Components")
            pca_loadings_plot_path = "pca_loadings_plot.png"
            plt.savefig(pca_loadings_plot_path)
            plt.close()

            # Step 5: Reverse PCA
            reconstructed_scaled_data = self.pca.inverse_transform(self.pca_result)
            reconstructed_data = self.scaler.inverse_transform(reconstructed_scaled_data)
            reconstructed_df = pd.DataFrame(
                reconstructed_data,
                columns=self.columns_to_process["numeric"]
            )
            self.reconstructed_data = reconstructed_df  # Save the reconstructed data

            # Return success message, plot paths, and reconstructed data
            return (
                f"PCA applied with {n_components} components.",
                cumulative_variance_plot_path,
                pca_loadings_plot_path,
                reconstructed_df.head()  # Preview of reconstructed data
            )
        except Exception as e:
            return str(e), None, None, None

    def show_correlation_matrix(self):
        if self.columns_to_process["numeric"]:
            print("Correlation Matrix:")
            correlation_matrix = self.data[self.columns_to_process["numeric"]].corr()
            print(correlation_matrix)
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
            plt.title("Correlation Matrix")
            plt.show()
            return correlation_matrix
        else:
            print("No numeric columns available for correlation matrix.")
            return "No numeric columns available for correlation matrix."


# Placeholder for the preprocessor object
preprocessor = None

# Function to load data and initialize the DataPreprocessor
def load_data(file):
    global preprocessor  # Access the global variable
    if file is not None:
        try:
            # Load the dataset
            data = pd.read_csv(file.name)
            print("Data loaded successfully.")

            # Initialize the preprocessor
            preprocessor = DataPreprocessor(data)
            preprocessor.load_data()
            preprocessor.identify_columns()

            # Prepare outputs for Gradio
            data_preview = data.head()  # Preview of the data
            numeric_columns = preprocessor.columns_to_process["numeric"]
            categorical_columns = preprocessor.columns_to_process["categorical"]

            return (
                data_preview,  # Data preview
                gr.update(choices=numeric_columns, value=numeric_columns),  # Numeric columns
                gr.update(choices=categorical_columns, value=categorical_columns)  # Categorical columns
            )
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame(), gr.update(choices=[], value=[]), gr.update(choices=[], value=[])
    return pd.DataFrame(), gr.update(choices=[], value=[]), gr.update(choices=[], value=[])


# Interface function mappings
def scale_columns(selected_columns):
    return preprocessor.scale_numeric_data(selected_columns)

def encode_columns(selected_columns):
    return preprocessor.label_encode_categorical_data(selected_columns)

def format_date_columns():
    return preprocessor.format_date_columns()

def drop_missing_rows():
    return preprocessor.drop_missing_rows()

def drop_empty_columns():
    return preprocessor.drop_empty_columns()

def fill_missing_values(numeric_strategy, categorical_strategy):
    return preprocessor.fill_missing_values(numeric_strategy=numeric_strategy, categorical_strategy=categorical_strategy)

def perform_pca_gradio():
    global preprocessor
    if preprocessor is None:
        return "Error: No dataset loaded. Please upload a dataset first.", None, None
    return preprocessor.perform_pca()

def show_correlation_matrix():
    if preprocessor.columns_to_process["numeric"]:
        correlation_matrix = preprocessor.data[preprocessor.columns_to_process["numeric"]].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Matrix")
        return fig
    else:
        return "No numeric columns available for correlation matrix."
def prepare_preprocessed_data():
        if preprocessor and preprocessor.data is not None:
            # Retain all columns (processed and unprocessed) from the original dataset
            final_data = preprocessor.data.copy()

            # Save the dataset to a CSV file
            file_path = "preprocessed_data.csv"
            final_data.to_csv(file_path, index=False)

            # Show a preview of the final dataset
            data_preview = final_data.head()  # First few rows of the dataset

            return file_path, data_preview
        else:
            return None, pd.DataFrame()  # Default response if no data is available

def perform_eda():
    """
    Perform EDA tasks:
    1. Dataset summary (numeric and categorical columns).
    2. Missing values analysis.
    3. Correlation matrix for numeric columns.
    4. Unique value counts for categorical columns.
    """
    results = {}

    # Dataset Summary
    try:
        results['summary'] = preprocessor.data.describe(include='all').transpose()
    except Exception as e:
        results['summary_error'] = f"Error generating summary: {e}"

    # Missing Values Analysis
    try:
        missing_values = preprocessor.data.isnull().sum()
        missing_percent = (missing_values / len(preprocessor.data)) * 100
        results['missing_values'] = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage': missing_percent
        }).sort_values(by='Missing Values', ascending=False)
    except Exception as e:
        results['missing_values_error'] = f"Error analyzing missing values: {e}"

    # Correlation Analysis
    if preprocessor.columns_to_process['numeric']:
        try:
            results['correlation_matrix'] = preprocessor.data[
                preprocessor.columns_to_process['numeric']
            ].corr()
        except Exception as e:
            results['correlation_error'] = f"Error calculating correlation matrix: {e}"
    else:
        results['correlation_error'] = "No numeric columns available for correlation matrix."

    # Unique Value Counts for Categorical Columns
    try:
        unique_counts = {
            col: preprocessor.data[col].nunique()
            for col in preprocessor.columns_to_process['categorical']
        }
        results['unique_values'] = pd.DataFrame({
            'Column': list(unique_counts.keys()),
            'Unique Values': list(unique_counts.values())
        }).sort_values(by='Unique Values', ascending=False)
    except Exception as e:
        results['unique_values_error'] = f"Error calculating unique values: {e}"

    # Prepare the EDA Report for Gradio
    eda_report = ""

    if 'summary' in results:
        eda_report += "\n### Dataset Summary:\n"
        eda_report += results['summary'].to_string()

    if 'missing_values' in results:
        eda_report += "\n\n### Missing Values Analysis:\n"
        eda_report += results['missing_values'].to_string()

    if 'correlation_matrix' in results:
        eda_report += "\n\n### Correlation Matrix:\n"
        eda_report += results['correlation_matrix'].to_string()

    if 'unique_values' in results:
        eda_report += "\n\n### Unique Value Counts (Categorical Columns):\n"
        eda_report += results['unique_values'].to_string()

    # Handle errors
    for key, value in results.items():
        if 'error' in key:
            eda_report += f"\n\n{key}: {value}"

    print(eda_report)  # For debugging/logging purposes
    return eda_report


iface = gr.Blocks()

with gr.Blocks() as iface:
    gr.Markdown("# Data Preprocessing Tool")

    with gr.Tab("Upload and Identify"):
        file_input = gr.File(label="Upload Dataset")
        data_preview = gr.DataFrame(label="Dataset Preview")
        numeric_dropdown = gr.Dropdown(label="Numeric Columns", multiselect=True, choices=[])
        categorical_dropdown = gr.Dropdown(label="Categorical Columns", multiselect=True, choices=[])

        # Ensure the upload event is set inside the Blocks context
        file_input.upload(
            load_data,
            inputs=file_input,
            outputs=[data_preview, numeric_dropdown, categorical_dropdown]
        )

    with gr.Tab("Preprocessing"):
        gr.Markdown("### Select Columns")
        # Dropdown components for column selection
        numeric_dropdown = gr.Dropdown(label="Numeric Columns", multiselect=True, choices=[])
        categorical_dropdown = gr.Dropdown(label="Categorical Columns", multiselect=True, choices=[])

        file_input.upload(
            load_data,
            inputs=file_input,
            outputs=[data_preview, numeric_dropdown, categorical_dropdown]
        )
        gr.Markdown("### Scale Numeric Columns")
        scale_button = gr.Button("Scale Selected Columns")
        scale_output = gr.Textbox(label="Scaling Output")
        scale_button.click(scale_columns, inputs=numeric_dropdown, outputs=scale_output)

        gr.Markdown("### Encode Categorical Columns")
        encode_button = gr.Button("Encode Selected Columns")
        encode_output = gr.Textbox(label="Encoding Output")
        encode_button.click(encode_columns, inputs=categorical_dropdown, outputs=encode_output)


        gr.Markdown("### Format Date Columns")
        date_button = gr.Button("Format Date Columns")
        date_output = gr.Textbox(label="Date Formatting Output")
        date_button.click(format_date_columns, inputs=None, outputs=date_output)

        gr.Markdown("### Drop Missing Rows")
        drop_missing_button = gr.Button("Drop Rows with Missing Values")
        drop_missing_output = gr.Textbox(label="Drop Missing Rows Output")
        drop_missing_button.click(drop_missing_rows, inputs=None, outputs=drop_missing_output)

        gr.Markdown("### Drop Empty Columns")
        drop_empty_button = gr.Button("Drop Empty Columns")
        drop_empty_output = gr.Textbox(label="Drop Empty Columns Output")
        drop_empty_button.click(drop_empty_columns, inputs=None, outputs=drop_empty_output)

        gr.Markdown("### Fill Missing Values")
        numeric_strategy = gr.Radio(label="Numeric Strategy", choices=["mean", "median"], value="mean")
        categorical_strategy = gr.Radio(label="Categorical Strategy", choices=["mode"], value="mode")
        fill_missing_button = gr.Button("Fill Missing Values")
        fill_missing_output = gr.Textbox(label="Fill Missing Values Output")
        fill_missing_button.click(fill_missing_values, inputs=[numeric_strategy, categorical_strategy], outputs=fill_missing_output)

    with gr.Tab("Advanced Analysis"):
        with gr.Tab("Advanced Analysis"):
            gr.Markdown("### Perform PCA Automatically")
            pca_button = gr.Button("Perform PCA")
            pca_output_text = gr.Textbox(label="PCA Output")
            cumulative_variance_plot = gr.Image(label="Cumulative Variance Plot")
            pca_loadings_plot = gr.Image(label="PCA Loadings Plot")
            reconstructed_data_preview = gr.DataFrame(label="Reconstructed Data Preview")


            def perform_pca_gradio():
                global preprocessor
                if preprocessor is None:
                    return "Error: No dataset loaded. Please upload a dataset first.", None, None, None
                return preprocessor.perform_pca()


            pca_button.click(
                perform_pca_gradio,
                inputs=None,
                outputs=[
                    pca_output_text,
                    cumulative_variance_plot,
                    pca_loadings_plot,
                    reconstructed_data_preview
                ]
            )

        gr.Markdown("### Show Correlation Matrix")
        correlation_button = gr.Button("Show Correlation Matrix")
        correlation_output = gr.Plot(label="Correlation Matrix")
        correlation_button.click(show_correlation_matrix, inputs=None, outputs=correlation_output)

    with gr.Tab("EDA"):
        gr.Markdown("### Perform Exploratory Data Analysis")
        eda_button = gr.Button("Perform EDA")
        eda_output = gr.Textbox(label="EDA Report", lines=20, interactive=False)
        eda_button.click(perform_eda, inputs=None, outputs=eda_output)

    with gr.Tab("Download Preprocessed Dataset"):
        gr.Markdown("### Download the Preprocessed Dataset")

        with gr.Tab("Download Preprocessed Dataset"):
            gr.Markdown("### Preview and Download the Preprocessed Dataset")

            # Button to generate the preprocessed dataset
            download_button = gr.Button("Generate Download Link")
            dataset_preview = gr.DataFrame(label="Dataset Preview")
            download_file = gr.File(label="Download Dataset")

            # Connect the button to the function
            download_button.click(
                prepare_preprocessed_data,
                inputs=None,
                outputs=[download_file, dataset_preview]
            )

    iface.launch()


