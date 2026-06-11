# 🎬 Netflix Titles — Data Science Homework 1

A complete data analysis pipeline on the Netflix Titles dataset, covering EDA, data cleaning, visualization, feature engineering, mutual information-based feature selection, and PCA.

---

## 📁 Dataset

**Source:** [Netflix Movies and TV Shows — Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows)

[Netflix](https://en.wikipedia.org/wiki/Netflix) is one of the most popular media and video streaming platforms in the world. As of mid-2021, they have over **8,000 movies and TV shows** on their platform and more than **200 million subscribers** globally.

This tabular dataset contains listings of all titles available on Netflix, with the following columns:

| Column | Description |
|--------|-------------|
| `show_id` | Unique identifier for each title |
| `type` | Movie or TV Show |
| `title` | Name of the title |
| `director` | Director(s) of the title |
| `cast` | Main cast members |
| `country` | Country/countries of production |
| `date_added` | Date the title was added to Netflix |
| `release_year` | Original release year |
| `rating` | Content rating (e.g. PG-13, TV-MA) |
| `duration` | Duration in minutes (movies) or seasons (TV shows) |
| `listed_in` | Genre(s) |
| `description` | Short synopsis |

**Place the dataset at:** `/data/netflix_titles.csv`

## 📊 What the Script Does

### 1. Exploratory Data Analysis
- Shape, dtypes, memory usage
- Missing value counts and percentages per column
- Value distributions for key categorical columns
- Release year range and content volume insights

### 2. Data Cleaning
- Remove duplicate `show_id` entries
- Fill missing `director`, `cast`, `country` with `"Unknown"`
- Fill missing `rating` with the column mode
- Parse `date_added` to datetime; fill gaps with median date
- Split `duration` into a numeric value (`duration_value`) and unit (`duration_unit`)
- Drop rows with logically invalid release years

### 3. Preprocessing
- Label-encode `type` and `rating`
- StandardScaler normalization of the numeric feature matrix

### 4. Visualizations (14 plots total)

| File | Chart Type | What it shows |
|------|-----------|---------------|
| `plot_01_pie.png` | Pie chart | Movies vs TV Shows split |
| `plot_02_boxplot.png` | Box plot | Duration distribution by type |
| `plot_03_line.png` | Multi-line chart | Titles added per year (2010–2021) |
| `plot_04_bar.png` | Bar chart | Top 10 producing countries |
| `plot_05_grouped_bar.png` | Grouped bar | Movies vs TV Shows per country |
| `plot_06_stacked_bar.png` | Stacked bar | Rating breakdown by content type |
| `plot_07_scatter.png` | Scatter plot | Release year vs movie duration |
| `plot_08_bubble.png` | Bubble chart | Genre count vs average duration |
| `plot_09_errorbars.png` | Error bar chart | Mean duration per rating with 95% CI |
| `plot_10_interactive_line.html` | Plotly interactive | Country content growth over time |
| `plot_11_interactive_bubble.html` | Plotly interactive | Rating vs duration bubble chart |
| `plot_12_mutual_info.png` | Horizontal bar | Mutual Information feature scores |
| `plot_13_pca_scree.png` | Scree plot | PCA explained variance per component |
| `plot_14_pca_2d.png` | Scatter (PCA) | 2D projection colored by content type |

### 5. Feature Engineering
New features derived from the raw data:

| Feature | Method | Description |
|---------|--------|-------------|
| `age_when_added` | Arithmetic | Years between release and Netflix addition |
| `log_duration` | Math transform | Log of duration to reduce right-skew |
| `era` | Binning | Release decade bucket (Classic → Recent) |
| `duration_bucket` | Binning | Short / Standard / Long / Epic (movies) |
| `month_added` | Datetime | Month the title was added |
| `is_q4_release` | Datetime | Whether title was added in Q4 (holiday season) |
| `num_genres` | Count | Number of genres listed |
| `cast_size` | Aggregation | Number of cast members listed |
| `num_countries` | Aggregation | Number of production countries |
| `director_title_count` | Aggregation | How many Netflix titles the director has |
| `is_collab_title` | Combination | Multi-country AND multi-genre flag |

### 6. Feature Selection — Mutual Information
Ranks all engineered features by their ability to predict content type (Movie vs TV Show). Duration-based features score highest, confirming that runtime is the strongest signal separating the two types.

### 7. Dimensionality Reduction — PCA
- Scree plot showing explained variance per component
- 2D projection showing partial class separability driven primarily by PC1 (duration)

---

## 🛠 Dependencies

```
Kagglehub
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn

```
