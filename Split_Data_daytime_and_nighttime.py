import pandas as pd
import re


file_path = '/Users/katerinaargyriparoni/data/Tallinn/LST_2024_clipped/results_summary_2024_tallinn.csv'
data = pd.read_csv(file_path, sep=',')


print("Columns in the CSV file:", data.columns)


if 'image' not in data.columns:
    raise KeyError("The column 'image' was not found in the CSV file. Available columns are:", data.columns)


def extract_hour(image_name):
    match = re.search(r'T(\d{2})(\d{2})(\d{2})', image_name)
    if match:
        return int(match.group(1))
    return None


def extract_datetime(image_name):
    date_match = re.search(r'_(\d{8})T(\d{2})(\d{2})(\d{2})', image_name)
    if date_match:
        # Μετατροπή ημερομηνίας στη μορφή YYYY-MM-DD
        date_str = f"{date_match.group(1)[:4]}-{date_match.group(1)[4:6]}-{date_match.group(1)[6:]}"
        # Ώρα στη μορφή HH:MM
        time_str = f"{date_match.group(2)}:{date_match.group(3)}"
        return date_str, time_str
    return None, None


data['hour'] = data['image'].apply(extract_hour)
data['date'], data['time'] = zip(*data['image'].apply(extract_datetime))


daytime_data = data[(data['hour'] >= 6) & (data['hour'] <= 17)]
nighttime_data = data[(data['hour'] >= 18) | (data['hour'] <= 5)]


daytime_path = '/Users/katerinaargyriparoni/data/Tallinn/LST_2020_clipped/results_summary_2024_tallinn_daytime.csv'
nighttime_path = '/Users/katerinaargyriparoni/data/Tallinn/LST_2020_clipped/results_summary_2024_tallinn_nighttime.csv'

daytime_data.to_csv(daytime_path, index=False)
nighttime_data.to_csv(nighttime_path, index=False)

print(f"Daytime data saved to {daytime_path}")
print(f"Nighttime data saved to {nighttime_path}")