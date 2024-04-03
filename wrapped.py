import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import re
from datetime import datetime, timedelta
import pytz

    

def load_data(folder_path, address):
    all_dfs = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path, filename))
            filtered_df = df[df['Voter'].str.lower() == address.lower()]
            all_dfs.append(filtered_df)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def create_cumulative_chart(final_df):
    final_df['Quarter'] = final_df['Tx Timestamp'].dt.to_period("Q")
    quarterly_data = final_df.groupby('Quarter').agg({'AmountUSD': np.sum}).reset_index()
    quarterly_data['CumulativeDonations'] = quarterly_data['AmountUSD'].cumsum()
    quarterly_data['Quarter'] = quarterly_data['Quarter'].dt.start_time

    fig = px.line(quarterly_data, x='Quarter', y='CumulativeDonations',
                  labels={'CumulativeDonations': 'Cumulative Donation Amount (USD)', 'Quarter': 'Quarter'},
                  markers=True)
    fig.update_layout(xaxis_title='Quarter', yaxis_title='Cumulative Donation Amount (USD)', yaxis=dict(range=[0, max(quarterly_data['CumulativeDonations'])*1.1]), width=800)
    fig.update_traces(line=dict(color='#00433B'))
    return fig

def create_heatmap(final_df):
    final_df['Year'] = final_df['Tx Timestamp'].dt.year.astype(int)
    final_df['Quarter'] = final_df['Tx Timestamp'].dt.quarter
    heatmap_data = final_df.groupby(['Year', 'Quarter']).agg({'AmountUSD': np.sum}).reset_index()
    heatmap_data_pivot = heatmap_data.pivot(index='Quarter', columns='Year', values='AmountUSD').fillna(0)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data_pivot.values,
        y='Q' + heatmap_data_pivot.index.astype(str),
        x=heatmap_data_pivot.columns.astype(str),
        hoverongaps=False,
        colorscale='YlOrRd'))
    fig.update_layout(
        title='Donation Heatmap by Quarter and Year',
        yaxis_nticks=36,
        yaxis_title='Quarter',
        xaxis_title='Year',
        xaxis=dict(tickmode='linear', type='category'),
        width=800)
    return fig

def display_donation_history(final_df):
    desired_columns = ['Project Name', 'AmountUSD', 'Tx Timestamp', 'Round Name']
    display_df = final_df[desired_columns].copy()
    display_df['AmountUSD'] = display_df['AmountUSD'].apply(lambda x: f"${x:,.2f}")
    display_df = display_df.sort_values('Tx Timestamp', ascending=False)
    return display_df

def display_top_projects_treemap(final_df):
    # Grouping by 'Project Name' and summing up 'AmountUSD'
    project_donations = final_df.groupby('Project Name').agg({'AmountUSD': 'sum'}).reset_index()
    
    # Sorting to get top projects by donation amount - though for a treemap, you might not need to limit to top 10
    top_projects = project_donations.sort_values('AmountUSD', ascending=False)

    # Creating a treemap
    custom_colors = ['#00433b','#c1eaff','#edfeda','#edfeda','#4fb8ef']
    fig = px.treemap(
        top_projects, 
        path=[px.Constant('All Projects'), 'Project Name'],  # Root node and branching
        values='AmountUSD',
        color_discrete_sequence=custom_colors
    )
    fig.update_layout(width=750, height=750)
    return fig

def update_for_cgrants_alpha(row, lookup_df):
    if row['Source'] in ['CGrants', 'Alpha']:
        # Filter lookup rows where 'Start Date' is before the 'Tx Timestamp'        
        valid_lookups = lookup_df[(lookup_df['Start Date'] <= row['Tx Timestamp']) & (lookup_df['End Date'] >= row['Tx Timestamp'])]
        
        if not valid_lookups.empty:
            # Find the closest 'Start Date'
            closest_row = valid_lookups.iloc[(valid_lookups['Start Date'] - row['Tx Timestamp']).abs().argsort()[:1]]
            return pd.Series([closest_row['Round Name'].values[0], closest_row['Aggregate Name'].values[0]])
    
    return pd.Series([row['Round Name'], None])  # Return original Round Name and None if no update

def update_for_grantsstack(row, lookup_df):
    if row['Source'] == 'GrantsStack':
        match = lookup_df[lookup_df['ID'] == row['Round Address']]
        if not match.empty:
            return pd.Series([match['Round Name'].values[0], match['Aggregate Name'].values[0]])
    
    return pd.Series([row['Round Name'], None])  # Return original Round Name and None if no update

def create_sunburst_chart(dataframe):

    
    # Create the sunburst chart
    fig = px.sunburst(
        dataframe,
        path=['Aggregate Name', 'Project Name'],  # Define hierarchy
        values='AmountUSD',
        color='AmountUSD',  # Optional: sectors colored based on 'AmountUSD'
        color_continuous_scale=[[0, '#C4F091'], [0.25, '#8EDA93'], [0.5, '#4A9A82'], [0.75, '#16626A'], [1, '#00433B']]
    )
    fig.update_layout(width=1000, height=1000)
    return fig

def get_recommendations(folder_path, voter):
    all_dfs = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path, filename))
            all_dfs.append(df)

    GG_round_addresses_df = pd.read_csv('./Gitcoin Wrapped/GS GG Rounds.csv')

    all_data = pd.concat(all_dfs, ignore_index=True)
    # Convert 'Voter' to lowercase for case-insensitive comparison
    df_lower = all_data.copy()
    df_lower['Voter'] = df_lower['Voter'].str.lower()
    df_lower['Tx Timestamp'] = pd.to_datetime(df_lower['Tx Timestamp'],format='mixed')

    # Convert both 'Round Address' in df_lower and 'ID' in known_round_addresses_df to lowercase for comparison
    df_lower['Round Address'] = df_lower['Round Address'].str.lower()
    GG_round_addresses_df['ID'] = GG_round_addresses_df['ID'].str.lower()

    # Now perform the isin check with both fields in lowercase
    df_lower = df_lower[df_lower['Round Address'].isin(GG_round_addresses_df['ID'])]

    #st.dataframe(GG_round_addresses_df)
    #st.dataframe(df_lower.head(10))

    voter_lower = voter.lower()

    # Step 1
    top_addresses = df_lower[df_lower['Voter'] == voter_lower].groupby('PayoutAddress').agg({'AmountUSD': 'sum'}).reset_index().sort_values('AmountUSD', ascending=False).head(3)
    #st.write("Step 1: Top Addresses")
    #st.dataframe(top_addresses)

    # Step 2: Modification here to extract just the list of voters
    # Define the date 12 months ago from the current time
    twelve_months_ago = datetime.now(pytz.utc) - timedelta(days=365)

    # Now perform the comparison
    recent_transactions = df_lower[df_lower['Tx Timestamp'] > twelve_months_ago]
    other_voters = recent_transactions[recent_transactions['PayoutAddress'].isin(top_addresses['PayoutAddress']) & (recent_transactions['Voter'] != voter_lower)]
    unique_other_voters = other_voters['Voter'].drop_duplicates()
    #st.write("Step 2: Other Voters in the Last 12 Months")
    #st.dataframe(unique_other_voters)

    # Proceeding with steps as they are if they're suitable for your context
    # Step 3
    # Filter the DataFrame to include only rows from unique other voters
    filtered_by_voters = df_lower[df_lower['Voter'].isin(unique_other_voters)]

    # Aggregate contributions by PayoutAddress among these voters and sort them
    top_supports_by_other_voters = filtered_by_voters.groupby('PayoutAddress') \
                                                     .agg({'AmountUSD': 'sum'}) \
                                                     .reset_index() \
                                                     .sort_values('AmountUSD', ascending=False)

    #st.write("Step 3: Top Payout Addresses Supported by Other Unique Voters")
    #st.dataframe(top_supports_by_other_voters.head(10))

    # Step 4
    voter_addresses = df_lower[df_lower['Voter'] == voter_lower]['PayoutAddress'].unique()
    recommended_addresses = top_supports_by_other_voters[~top_supports_by_other_voters['PayoutAddress'].isin(voter_addresses)].head(10)
    #st.write("Step 4: Recommended Addresses")
    #st.dataframe(recommended_addresses)

    # Step 5
    df_unique = df_lower.drop_duplicates(subset='PayoutAddress')
    recommended_projects = recommended_addresses.merge(df_unique[['PayoutAddress', 'Project Name']].drop_duplicates(), on='PayoutAddress', how='left')
    #st.write("Step 5: Recommended Projects")
    #st.dataframe(recommended_projects['Project Name'])

    return recommended_projects[['Project Name']]

# Main function to orchestrate the workflow
def main():
    st.set_page_config(layout='wide')
    tcol1,tcol2,tcol3 = st.columns([1,3,1])
    
    tcol2.image("https://i.postimg.cc/wB32R5J1/Gbanner.png")
    tcol2.title('Gitcoin Grants Impact Dasboard')
    tcol2.markdown('### Your support for Gitcoin Grants has a story. \n ### Let\'s reveal it together.')
        
    folder_path = './Gitcoin Wrapped/data'
    lookup_df = pd.read_csv('./Gitcoin Wrapped/GS GG Rounds.csv')
    lookup_df['Start Date'] = pd.to_datetime(lookup_df['Start Date']).dt.tz_localize(None)
    lookup_df['End Date'] = pd.to_datetime(lookup_df['End Date']).dt.tz_localize(None)


    address = tcol2.text_input('Enter your Ethereum address below to uncover your unique impact story (starting "0x"):', help='ENS not supported, please enter 42-character hexadecimal address starting with "0x"')

    if address and address != 'None':
        my_bar = tcol2.progress(0, text='Looking up! Please wait.')
        if not re.match(r'^(0x)?[0-9a-f]{40}$', address, flags=re.IGNORECASE):
            tcol2.error('Not a valid address. Please enter a valid 42-character hexadecimal Ethereum address starting with "0x"')
            my_bar.empty()
        else:
            my_bar.progress(10, "Valid address found. Searching your contributions...brb.")

            # Get all contributions associated with the address
            all_df = load_data(folder_path, address)
            
            if not all_df.empty:
                #final_df['Tx Timestamp'] = pd.to_datetime(final_df['Tx Timestamp'], format='mixed')
                all_df['Tx Timestamp'] = pd.to_datetime(all_df['Tx Timestamp'], format='mixed').dt.tz_localize(None)

                # Rationalize round names
                all_df[['Round Name', 'Aggregate Name']] = all_df.apply(lambda row: update_for_cgrants_alpha(row, lookup_df) if row['Source'] in ['CGrants', 'Alpha'] else update_for_grantsstack(row, lookup_df), axis=1)
                # Adding the 'GG' column to indentify if the contribution is for a Gitcoin Grant
                all_df['GG'] = all_df['Aggregate Name'].apply(lambda x: 'N' if pd.isna(x) or x == '' else 'Y')

                final_df = all_df[all_df['GG'] == 'Y']

                if not final_df.empty:
                    # Display Key Stats
                    earliest_tx_timestamp = final_df['Tx Timestamp'].min()
                    sum_amount_usd = final_df['AmountUSD'].sum()
                    num_rows = final_df.shape[0]
                    unique_payout_addresses = final_df['PayoutAddress'].nunique()

                    qcol1, qcol2, qcol3, qcol4, qcol5, qcol6 = st.columns([1,2,2,2,2,1])    
                    with qcol2:
                        st.metric(label="Your Gitcoin Grants debut was on", value=earliest_tx_timestamp.strftime('%d-%b-%Y'))
                    with qcol3:
                        st.metric(label="Grantees you have empowered", value=unique_payout_addresses)
                    with qcol4:
                        st.metric(label="Your total impact", value="${:,.0f}".format(sum_amount_usd))
                    with qcol5:    
                        st.metric(label="Your contribution count", value=num_rows)                        
                    
                    tcol1,tcol2,tcol3 = st.columns([1,3,1])
                    # Display Treemap of Projects
                    with tcol2:
                        st.plotly_chart(create_cumulative_chart(final_df))
    
                        #st.plotly_chart(create_heatmap(final_df))

                        st.plotly_chart(display_top_projects_treemap(final_df), use_container_width=True)
                        st.plotly_chart(create_sunburst_chart(final_df), use_container_width=True)

                        # Display Donation History    
                    
                        st.markdown("#### Donation History")
                        display_df = display_donation_history(final_df)
                        st.dataframe(display_df, hide_index=True)
                        donation_csv = display_df.to_csv(index=False)
                        st.download_button(
                            label="Download data as CSV",
                            data=donation_csv,
                            file_name='data.csv',
                            mime='text/csv',
                        )

                        # Filter the DataFrame for rows where 'GG' is 'N'
                        st.markdown("#### Your Top 5 Rounds outside Gitcoin Grants")
                        grouped_df = all_df.groupby(['Round Name', 'GG'])['AmountUSD'].sum().reset_index()

                        not_gg_df = grouped_df[grouped_df['GG'] == 'N']

                        # Select the top 5 entries based on 'Round Name' and 'AmountUSD'
                        top_5_rounds = not_gg_df.sort_values('AmountUSD', ascending=False).head(5)
                        top_5_rounds = top_5_rounds.drop(columns='GG')
                        st.dataframe(top_5_rounds,hide_index=True)

                        my_bar.progress(70,"🫡 thank you for your support to Gitcoin Grants. Check out your stats below while we build your personalized recommendations list for future rounds!")
                        # Recommendations
                        recommendations = get_recommendations(folder_path, address)
                        st.dataframe(recommendations, hide_index=True)

                        my_bar.empty()

                else:
                    my_bar.empty()
                    tcol2.write("While there are contributions made from this address on Grants Stack, no contribution data found for Gitcoin Grants.")                    
            else:
                my_bar.empty()
                tcol2.write("No contribution data available for this address.")

if __name__ == "__main__":
    main()
