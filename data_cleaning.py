import pandas as pd

# Load the main dataset with Refugees data
refugee_df = pd.read_csv('datasets/UNHCR_forcibly_displaced_population.csv')

# Sum up all kind of forcibly displaced population in a Refugees column
population_columns = [
                      'Refugees under UNHCRs mandate',
                      'Asylum-seekers',
                      'Other people in need of international protection',
                      'Others of concern']
refugee_df['Refugees'] = refugee_df[population_columns].sum(axis=1)
refugee_df.drop(population_columns, inplace = True, axis=1)

# Drop the rows with less then 1000 refugees, internal migrants, stateless refugees and unknow countries
refugee_df = refugee_df.drop(refugee_df[
    (refugee_df['Refugees'] < 1000) | 
    (refugee_df['Country of origin (ISO)'] == refugee_df['Country of asylum (ISO)']) |
    (refugee_df['Country of origin (ISO)'] == 'XXA') |
    (refugee_df['Country of asylum (ISO)'] == 'XXA') |
    (refugee_df['Country of origin (ISO)'] == 'UNK') |
    (refugee_df['Country of asylum (ISO)'] == 'UNK')].index)

# Merge with World Region dataset
region_df = pd.read_csv('datasets/UNStats_standard_country_region.csv')
refugee_df = refugee_df.merge(
    region_df[['Sub-region Name','Intermediate Region Name','ISO-alpha3 Code']], 
    how='left', 
    left_on=['Country of asylum (ISO)'], 
    right_on=['ISO-alpha3 Code'])
refugee_df['Asylum Region'] = refugee_df.apply(lambda row: row['Sub-region Name'] if pd.isna(row['Intermediate Region Name']) else row['Intermediate Region Name'] , axis=1)
refugee_df.drop(['Sub-region Name','Intermediate Region Name','ISO-alpha3 Code'], inplace = True, axis=1)

# Merge with the Geographic Distance dataset
geo_df = pd.read_csv('datasets/CEPII_geographic_distances.csv')
refugee_df = refugee_df.merge(
    geo_df[['iso_o','iso_d','distw']], 
    how='left', 
    left_on=['Country of origin (ISO)', 'Country of asylum (ISO)'], 
    right_on=['iso_o', 'iso_d'])
refugee_df.drop(['iso_o','iso_d'], inplace = True, axis=1)

# Prepare the column Distance (Km) as float64 data
refugee_df = refugee_df.rename(columns={'distw': 'Distance (Km)'})
refugee_df['Distance (Km)'] = refugee_df.apply(lambda row: row['Distance (Km)'].replace(",", "."), axis=1)
refugee_df['Distance (Km)'] = pd.to_numeric(refugee_df['Distance (Km)'])

# Dataset with Human Development Indices
dev_df = pd.read_csv('datasets/UNDP_Composite_indices_complete_1990_2021.csv')
dev_columns = ['iso3',
                'hdi_2012','hdi_2013','hdi_2014','hdi_2015','hdi_2016',
                'hdi_2017','hdi_2018','hdi_2019','hdi_2020','hdi_2021',
                'le_2012','le_2013','le_2014','le_2015','le_2016',
                'le_2017','le_2018','le_2019','le_2020','le_2021',
                'eys_2012','eys_2013','eys_2014','eys_2015','eys_2016',
                'eys_2017','eys_2018','eys_2019','eys_2020','eys_2021',
                'mys_2012','mys_2013','mys_2014','mys_2015','mys_2016',
                'mys_2017','mys_2018','mys_2019','mys_2020','mys_2021',
                'gnipc_2012','gnipc_2013','gnipc_2014','gnipc_2015','gnipc_2016',
                'gnipc_2017','gnipc_2018','gnipc_2019','gnipc_2020','gnipc_2021',
                'gdi_2012','gdi_2013','gdi_2014','gdi_2015','gdi_2016',
                'gdi_2017','gdi_2018','gdi_2019','gdi_2020','gdi_2021',
                'gii_2012','gii_2013','gii_2014','gii_2015','gii_2016',
                'gii_2017','gii_2018','gii_2019','gii_2020','gii_2021',
                'phdi_2012','phdi_2013','phdi_2014','phdi_2015','phdi_2016',
                'phdi_2017','phdi_2018','phdi_2019','phdi_2020','phdi_2021']

# Merge with Human Developement Indices for the Country of Origin
refugee_df = refugee_df.merge(
                              dev_df[dev_columns], 
                              how='left', 
                              left_on=['Country of origin (ISO)'], 
                              right_on=['iso3'])
refugee_df['HDI origin'] = refugee_df.apply(lambda row: row['hdi_' + str(row.Year)], axis=1)
refugee_df['LE origin'] = refugee_df.apply(lambda row: row['le_' + str(row.Year)], axis=1)
refugee_df['EYS origin'] = refugee_df.apply(lambda row: row['eys_' + str(row.Year)], axis=1)
refugee_df['MYS origin'] = refugee_df.apply(lambda row: row['mys_' + str(row.Year)], axis=1)
refugee_df['GNIPC origin'] = refugee_df.apply(lambda row: row['gnipc_' + str(row.Year)], axis=1)
refugee_df['GDI origin'] = refugee_df.apply(lambda row: row['gdi_' + str(row.Year)], axis=1)
refugee_df['GII origin'] = refugee_df.apply(lambda row: row['gii_' + str(row.Year)], axis=1)
refugee_df['PHDI origin'] = refugee_df.apply(lambda row: row['phdi_' + str(row.Year)], axis=1)

# Drop the columns from dev_df
refugee_df.drop(dev_columns, inplace = True, axis=1)

# Merge with Human Developement Indices for the Country of Asylum
refugee_df = refugee_df.merge(
                              dev_df[dev_columns], 
                              how='left', 
                              left_on=['Country of asylum (ISO)'], 
                              right_on=['iso3'])
refugee_df['HDI asylum'] = refugee_df.apply(lambda row: row['hdi_' + str(row.Year)], axis=1)
refugee_df['LE asylum'] = refugee_df.apply(lambda row: row['le_' + str(row.Year)], axis=1)
refugee_df['EYS asylum'] = refugee_df.apply(lambda row: row['eys_' + str(row.Year)], axis=1)
refugee_df['MYS asylum'] = refugee_df.apply(lambda row: row['mys_' + str(row.Year)], axis=1)
refugee_df['GNIPC asylum'] = refugee_df.apply(lambda row: row['gnipc_' + str(row.Year)], axis=1)
refugee_df['GDI asylum'] = refugee_df.apply(lambda row: row['gdi_' + str(row.Year)], axis=1)
refugee_df['GII asylum'] = refugee_df.apply(lambda row: row['gii_' + str(row.Year)], axis=1)
refugee_df['PHDI asylum'] = refugee_df.apply(lambda row: row['phdi_' + str(row.Year)], axis=1)

# Drop the columns from dev_df
refugee_df.drop(dev_columns, inplace = True, axis=1)

# Add diff columns between Origin and Asylum
refugee_df['HDI diff'] = refugee_df.apply(lambda row: row['HDI asylum'] - row['HDI origin'], axis=1)
refugee_df['LE diff'] = refugee_df.apply(lambda row: row['LE asylum'] - row['LE origin'], axis=1)
refugee_df['EYS diff'] = refugee_df.apply(lambda row: row['EYS asylum'] - row['EYS origin'], axis=1)
refugee_df['MYS diff'] = refugee_df.apply(lambda row: row['MYS asylum'] - row['MYS origin'], axis=1)
refugee_df['GNIPC diff'] = refugee_df.apply(lambda row: row['GNIPC asylum'] - row['GNIPC origin'], axis=1)
refugee_df['GDI diff'] = refugee_df.apply(lambda row: row['GDI asylum'] - row['GDI origin'], axis=1)
refugee_df['GII diff'] = refugee_df.apply(lambda row: row['GII asylum'] - row['GII origin'], axis=1)
refugee_df['PHDI diff'] = refugee_df.apply(lambda row: row['PHDI asylum'] - row['PHDI origin'], axis=1)

# Drop data related to Origin Country and unused columns
refugee_df.drop(['Country of origin (ISO)', 
                 'Country of asylum (ISO)',
                 'HDI origin','LE origin',
                 'EYS origin',
                 'MYS origin',
                 'GNIPC origin',
                 'GDI origin',
                 'GII origin',
                 'PHDI origin'], inplace = True, axis=1)

# Check for Missing Values
missing_values_count = 0
for i in range(len(refugee_df.index)) :
    n = refugee_df.iloc[i].isnull().sum()
    if (n > 0):
        missing_values_count += 1

print("Missing Values Count: " + str(missing_values_count))

# Drop all rows whith missing data
refugee_df = refugee_df.dropna()
refugee_df.to_csv('datasets/data.csv', index=False)

# Print final information
print(refugee_df.tail())
print(refugee_df.dtypes)
print("Final dataset length without missing values: " + str(len(refugee_df.index)))
