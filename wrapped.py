import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import re
from datetime import datetime, timedelta
import pytz
import streamlit.components.v1 as components
import psycopg2 as pg

# Now you can use os.getenv to access your variables
db_host= st.secrets['DB_HOST']
db_port = st.secrets['DB_PORT']
db_name = st.secrets['DB_NAME']
db_username = st.secrets['DB_USERNAME']
db_password = st.secrets['DB_PASSWORD']

indexer_db_host= st.secrets['INDEXER_DB_HOST']
indexer_db_port = st.secrets['INDEXER_DB_PORT']
indexer_db_name = st.secrets['INDEXER_DB_NAME']
indexer_db_username = st.secrets['INDEXER_DB_USERNAME']
indexer_db_password = st.secrets['INDEXER_DB_PASSWORD']

def load_data(address):


    # Gets GG1 through GG19 data
    query_1 = """
    SELECT d."round_num" AS "Round Num",
           d."round_name" AS "Round Name",
           d."donor_address" AS "Voter",
           d."amount_in_usd" AS "AmountUSD",
           d."recipient_address" AS "PayoutAddress",
           d."timestamp" AS "Tx Timestamp",
           d."project_name" AS "Project Name",
           d."source" AS "Source"
    FROM all_donations d
    WHERE lower(d."donor_address") = lower(%s)
    """

    # Gets GG20 data
    query_2 = """
    SELECT
        "chain_data_65"."donations"."round_id" AS "Round Num",
        ("chain_data_65"."rounds"."round_metadata" #>> array [ 'name' ] :: text [ ]) :: text AS "Round Name",
        "chain_data_65"."donations"."donor_address" AS "Voter",
        "chain_data_65"."donations"."amount_in_usd" AS "AmountUSD",
         "chain_data_65"."donations"."recipient_address" AS "PayoutAddress",
        "chain_data_65"."donations"."timestamp" as "Tx Timestamp",
        ("chain_data_65"."applications"."metadata"#>>array [ 'application','project','title' ]::text [])::text AS "Project Name",
        "chain_data_65"."rounds"."id" AS "Round Address",
        'GrantsStack' AS "Source"
    FROM
        "chain_data_65"."rounds",
        "chain_data_65"."applications",
        "chain_data_65"."donations"
    where
        -- filter for GG20 rounds
        (
            ("chain_data_65"."rounds"."chain_id" = '42161' AND "chain_data_65"."rounds"."id"  IN ('23','24','25','26','27','28','29','31'))
        or
            ("chain_data_65"."rounds"."chain_id" = '10' AND "chain_data_65"."rounds"."id"  IN ('9'))
        )
        AND
        -- Join applications and rounds
        "chain_data_65"."applications"."round_id" = "chain_data_65"."rounds"."id" AND 
        "chain_data_65"."applications"."chain_id" = "chain_data_65"."rounds"."chain_id" AND
        -- Join applications and donations
        "chain_data_65"."applications"."chain_id"  = "chain_data_65"."donations"."chain_id" AND
        "chain_data_65"."applications"."round_id"  = "chain_data_65"."donations"."round_id" AND 
        "chain_data_65"."applications"."id"  = "chain_data_65"."donations"."application_id" AND
        -- Filter on user's address
        lower("chain_data_65"."donations"."donor_address") = lower(%s)
    """

    # Connect to the PostgreSQL database
    conn = pg.connect(host=db_host, port=db_port, dbname=db_name, user=db_username, password=db_password)
    # indexer_conn = pg.connect(host=indexer_db_host, port=indexer_db_port, dbname=indexer_db_name, user=indexer_db_username, password=indexer_db_password)
    all_dfs = pd.read_sql_query(query_1, conn, params=(address.lower(),))
    # all_dfs_2 = pd.read_sql_query(query_2, indexer_conn, params=(address.lower(),))

    # all_dfs = pd.concat([all_dfs_1, all_dfs_2], ignore_index=True)
    
    conn.close()

    return all_dfs if not all_dfs.empty else pd.DataFrame()

def create_cumulative_chart(final_df):
    final_df['Quarter'] = final_df['Tx Timestamp'].dt.to_period("Q")
    quarterly_data = final_df.groupby('Quarter').agg({'AmountUSD': np.sum}).reset_index()
    quarterly_data['CumulativeDonations'] = quarterly_data['AmountUSD'].cumsum()
    quarterly_data['Quarter'] = quarterly_data['Quarter'].dt.start_time

    # Format the CumulativeDonations for display: round to the nearest number and convert to currency format
    quarterly_data['CumulativeDonationsText'] = quarterly_data['CumulativeDonations'].apply(lambda x: f"${x:,.0f}")

    fig = px.line(quarterly_data, x='Quarter', y='CumulativeDonations',
                  labels={'CumulativeDonations': 'Cumulative Donation Amount (USD)', 'Quarter': 'Quarter'},
                  markers=True,
                  text='CumulativeDonationsText')  # Add text labels for each point

    # Update layout and traces
    fig.update_layout(xaxis_title='Quarter', yaxis_title='Cumulative Donation Amount (USD)',
                      yaxis=dict(range=[0, max(quarterly_data['CumulativeDonations'])*1.1]), width=800)
    
    # Positioning the text labels on the top left of each marker
    fig.update_traces(textposition='top left', line=dict(color='#00433B'))

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
    display_df = display_df.sort_values('Tx Timestamp', ascending=False, na_position='first')
    return display_df

def display_top_projects_treemap(final_df):
    # Grouping by 'Project Name' and summing up 'AmountUSD'
    project_donations = final_df.groupby('Project Name').agg({'AmountUSD': 'sum'}).reset_index()
    
    # Adding a new column that combines 'Project Name' and 'AmountUSD' for display
    project_donations['Label'] = project_donations.apply(lambda row: f"{row['Project Name']} (${row['AmountUSD']:,.2f})", axis=1)
    top_projects = project_donations.sort_values('AmountUSD', ascending=False)

    # Creating a treemap
    custom_colors = ['#00433b', '#c1eaff', '#edfeda', '#4fb8ef']
    fig = px.treemap(
        top_projects, 
        path=[px.Constant('All Projects'), 'Label'],  # Use the new 'Label' for path to show both name and amount
        values='AmountUSD',
        color_discrete_sequence=custom_colors
    )

    fig.update_layout(width=750, height=750)
    return fig

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


def get_recommendations_gg20(df,address):
    
    # Find participating project in GG20
    query = """
        SELECT      
            CONCAT('https://explorer.gitcoin.co/#/round/',"chain_data_65"."rounds"."chain_id",'/',"chain_data_65"."rounds"."id",'/',"chain_data_65"."applications"."id") as Link,
            ("chain_data_65"."applications"."metadata"#>>array [ 'application','project','title' ]::text [])::text AS "Project Name",
            ("chain_data_65"."rounds"."round_metadata" #>> array [ 'name' ] :: text [ ]) :: text AS "Round Name",
            "chain_data_65"."rounds"."chain_id" as "Chain ID",
            "chain_data_65"."rounds"."id" as "Round ID",
            "chain_data_65"."applications"."id" as "Application ID",
            lower(("chain_data_65"."applications"."metadata"#>>array [ 'application','recipient' ]::text [])::text) AS "PayoutAddress",
            CASE 
                WHEN "chain_data_65"."donations"."donor_address" IS NOT NULL
                THEN 'Yes'
            ELSE 'No'
            END AS "Donated"
        FROM
            "chain_data_65"."rounds" 
            JOIN
            "chain_data_65"."applications" 
        ON
            "chain_data_65"."applications"."round_id" = "chain_data_65"."rounds"."id" AND 
            "chain_data_65"."applications"."chain_id" = "chain_data_65"."rounds"."chain_id" AND
            "chain_data_65"."applications"."status" = 'APPROVED'
        LEFT OUTER JOIN
            "chain_data_65"."donations"
        ON
            "chain_data_65"."applications"."chain_id"  = "chain_data_65"."donations"."chain_id" AND
            "chain_data_65"."applications"."round_id"  = "chain_data_65"."donations"."round_id" AND 
            "chain_data_65"."applications"."id"  = "chain_data_65"."donations"."application_id" AND
            lower("chain_data_65"."donations"."donor_address") = lower(%s)
        WHERE    
            -- filter for GG20 rounds
            (
                ("chain_data_65"."rounds"."chain_id" = '42161' AND "chain_data_65"."rounds"."id"  IN ('23','24','25','26','27','28','29','31'))
            or
                ("chain_data_65"."rounds"."chain_id" = '10' AND "chain_data_65"."rounds"."id"  IN ('9'))
            )
      """      

    indexer_conn = pg.connect(host=indexer_db_host, port=indexer_db_port, dbname=indexer_db_name, user=indexer_db_username, password=indexer_db_password)
    gg20_df = pd.read_sql_query(query, indexer_conn, params=(address.lower(),))

    #mask = gg20_df['PayoutAddress'].isin(df['PayoutAddress'])
    #filtered_gg20_df = gg20_df[mask]

    # Merge gg20 data with user's donation history
    merged_df = pd.merge(gg20_df, df[['PayoutAddress', 'AmountUSD']], on='PayoutAddress', how='inner')
    
    # Sort the merged DataFrame based on the AmountUSD column
    sorted_merged_df = merged_df.sort_values(by='AmountUSD', ascending=False)
    
    return sorted_merged_df

# Main function to orchestrate the workflow
def main():
    # Set the columns for display
    st.set_page_config(layout='wide')
    tcol1,tcol2,tcol3 = st.columns([1,3,1])

    # Set image title and header
    tcol2.image("https://i.postimg.cc/wB32R5J1/Gbanner.png")
    tcol2.title('Gitcoin Grants Impact Dashboard')
    tcol2.markdown('### Your support for Gitcoin Grants has a story. Let\'s reveal it together.')

    # Load the configuration file (GS GG Rounds) that identifies what rounds to include in the statistics
    folder_path = './data'
    lookup_df = pd.read_csv('./GS GG Rounds.csv')
    lookup_df['Start Date'] = pd.to_datetime(lookup_df['Start Date']).dt.tz_localize(None)
    lookup_df['End Date'] = pd.to_datetime(lookup_df['End Date']).dt.tz_localize(None)
    lookup_df['ID'] = lookup_df['ID'].str.lower()
    
    # Initialize session state for address if not already set
    if 'address' not in st.session_state:
        st.session_state.address = None
    
    # Check if the address is provided in the URL
    query_params = st.query_params.get_all('address')
    if len(query_params) == 1 and not st.session_state.address:
        # Set the initial address from the URL in session state
        st.session_state.address = query_params[0]
    
    # Create an input field for the address - it uses the session state address or updates it
    address_input = tcol2.text_input(
        'Enter your Ethereum address below to uncover your unique impact story (starting "0x"):', 
        value = st.session_state.address or '',
        key='address',
        help='ENS not supported, please enter 42-character hexadecimal address starting with "0x"'
        )
    
    # Now, use the address from the session state for further processing
    address = st.session_state.address

    if address and address != 'None':
        my_bar = tcol2.progress(0, text='Looking up! Please wait.')
        
        # Validate the syntax for the address
        if not re.match(r'^(0x)?[0-9a-f]{40}$', address, flags=re.IGNORECASE):
            tcol2.error('Not a valid address. Please enter a valid 42-character hexadecimal Ethereum address starting with "0x"')
            my_bar.empty()
        else:
            my_bar.progress(10, "Valid address found. Searching your contributions...brb.")

            # Get all contributions associated with the address
            all_df = load_data(address)
            
            if not all_df.empty:
                #final_df['Tx Timestamp'] = pd.to_datetime(final_df['Tx Timestamp'], format='mixed')
                all_df['Tx Timestamp'] = pd.to_datetime(all_df['Tx Timestamp'], format='mixed').dt.tz_localize(None)

                # Rationalize round names
                # all_df[['Round Name', 'Aggregate Name']] = all_df.apply(lambda row: update_for_cgrants_alpha(row, lookup_df) if row['Source'] in ['CGrants', 'Alpha'] else update_for_grantsstack(row, lookup_df), axis=1)
                # Adding the 'GG' column to identify if the contribution is for a Gitcoin Grant
                #all_df['GG'] = all_df['Aggregate Name'].apply(lambda x: 'N' if pd.isna(x) or x == '' else 'Y')
                all_df['GG'] = all_df['Round Num'].apply(lambda x: 'Y' if pd.notna(x) else 'N')

                # Due to missing Tx Timestamp on GG20, default to day 1 for cumulative dashboard reporting
                # default_date = pd.Timestamp('2024-04-23')
                # mask = (all_df['Tx Timestamp'].isna()) & (all_df['Aggregate Name'] == 'GG20')
                # all_df.loc[mask, 'Tx Timestamp'] = default_date

                final_df = all_df[all_df['GG'] == 'Y']
                not_ggrant_df = all_df[all_df['GG'] == 'N']


                if not final_df.empty:
                    # Display Key Stats
                    earliest_tx_timestamp = final_df['Tx Timestamp'].min()
                    sum_amount_usd = final_df['AmountUSD'].sum()
                    sum_amount_usd_not_ggrant = not_ggrant_df['AmountUSD'].sum()
                    num_rows = final_df.shape[0]
                    unique_payout_addresses = final_df['PayoutAddress'].nunique()

                    st.balloons()
                    tcol2.markdown("#")
                    tcol2.success("### Impact Overview - Your Contribution Snapshot")
                    tcol2.markdown("We are grateful to have you with us in this journey. \
                                    Each contribution you've made fuels the collective vision. \
                                    Here's a glimpse into the impact you've created:")

                    qcol1, qcol2, qcol3, qcol4 = st.columns([2,3,3,2])    
                    
                    with qcol2:
                        cont1 = st.container(border=True)
                        cont1.metric(label="Your Gitcoin Grants debut was on", value=earliest_tx_timestamp.strftime('%d-%b-%Y'))
                    with qcol3:
                        cont2 = st.container(border=True)
                        cont2.metric(label="Grantees you have empowered", value=unique_payout_addresses)
                    with qcol2:
                        cont3 = st.container(border=True)
                        cont3.metric(label="Your total impact", value="${:,.0f}".format(sum_amount_usd))
                    with qcol3:    
                        cont4 = st.container(border=True)
                        cont4.metric(label="Your contribution count", value=num_rows)                        
                    
                    tcol1,tcol2,tcol3 = st.columns([1,3,1])

                    if sum_amount_usd_not_ggrant > 0:
                        tcol2.caption('ℹ️ In addition to your contributions to Program and Community Rounds in Gitcoin Grants, you have donated $' + str(round(sum_amount_usd_not_ggrant,0)) + \
                                   ' to Independent Rounds on Grants Stack.')


                    # Social Sharing
                    
                    custom_url = f"https://ggwrapped.gitcoin.co/?address={address}"

                    tcol2.markdown("🌟 **Share Your Impact!** 🌍 Let the world know how you've contributed to the open-source ecosystem and beyond with Gitcoin Grants. Inspire others with your journey! 💫")
                    # Insert the custom URL into the HTML string using string formatting
                    html_content = f"""
                        <a href="https://twitter.com/share?ref_src=twsrc%5Etfw" class="twitter-share-button" 
                        data-text="I\'ve been a part of curating and signaling the next generation of high-impact projects through @gitcoin Grants 🫡 Join me and share your impact today! Check out my #ggwrapped Impact Dashboard " 
                        data-url="{custom_url}"
                        data-show-count="false"
                        data-size="Large" 
                        height=10
                        Tweet
                        </a>
                        <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    """
                    
                    with tcol2:
                        components.html(html_content)

                    
                    # Display Treemap of Projects
                    with tcol2:
                        st.markdown("#")
                        st.success("### Contribution Timeline - Evolution of Your Impact")
                        st.caption("To download the chart as an image, hover over the chart and click 📷 icon on top right")
                        st.plotly_chart(create_cumulative_chart(final_df))

                        #st.plotly_chart(create_heatmap(final_df))
                        st.success("### Support Landscape - Grantees You Backed")
                        st.caption("To download the chart as an image, hover over the chart and click 📷 icon on top right")
                        st.plotly_chart(display_top_projects_treemap(final_df), use_container_width=True)
                        
                        st.success("### Contribution Burst - Exploration of Your Impact")
                        st.caption("Click on any round to drill-down and drill-up. To download the chart as an image, hover over the chart and click 📷 icon on top right")
                        st.plotly_chart(create_sunburst_chart(final_df), use_container_width=True)

                        # Display Donation History    
                        st.success("### Journey of Giving - Your Detailed Donation History")
                        st.caption("Click on the download button to save as .csv file")
                        display_df = display_donation_history(final_df)
                        st.dataframe(display_df[['Project Name','AmountUSD','Round Name']], hide_index=True, use_container_width=True)
                        donation_csv = display_df.to_csv(index=False)
                        st.download_button(
                            label="Download data as CSV",
                            data=donation_csv,
                            file_name='data.csv',
                            mime='text/csv',
                            type="primary"
                        )

                        # Filter the DataFrame for rows where 'GG' is 'N'
                        st.markdown("#")
                        st.success("### Top Independent Rounds You Have Supported")
                        st.caption("Gitcoin 2.0 allows any EVM community to allocate capital in a transparent and democratic way.\
                        These are the top rounds you contributed to outside of Gitcoin Grants. \
                        To learn more about Gitcoin 2.0, read our whitepaper [here](https://www.gitcoin.co/whitepaper)")
                        grouped_df = all_df.groupby(['Round Name', 'GG'])['AmountUSD'].sum().reset_index()
                        not_gg_df = grouped_df[grouped_df['GG'] == 'N']

                        # Select the top 5 entries based on 'Round Name' and 'AmountUSD'
                        top_5_rounds = not_gg_df.sort_values('AmountUSD', ascending=False).head(5)
                        top_5_rounds = top_5_rounds.drop(columns='GG')
                        top_5_rounds['AmountUSD'] = top_5_rounds['AmountUSD'].apply(lambda x: f"${x:,.2f}")
                        st.dataframe(top_5_rounds,hide_index=True, use_container_width=True)
                        
                        #my_bar.progress(70,"🫡 thank you for your support to Gitcoin Grants. Check out your stats below while we build your personalized recommendations list for future rounds!")
                        # Recommendations
                        #recommendations = get_recommendations(folder_path, address)
                        #st.markdown("#")
                        #st.success("### Curated Opportunities: Your Next Potential Grantees")
                        #st.caption("We pulled a list of recommended grantees for you based on contributors' choices who support the projects you support the most.")
                        #st.caption("The projects listed below have received the most support from donors over the last 12 months, who also contributed to the top three projects you have most supported.")
                        #st.dataframe(recommendations, hide_index=True, use_container_width=True)

                        # Show favorite projects participating in GG20
                        st.markdown("#")
                        st.success("### Rediscover Your Favorites in GG20")
                        st.caption("Below is a list of projects you've previously supported and are participating in GG20. \
                            Consider showing your support again! For projects you've recently backed in GG20, you'll see a ✅.")
                            
                        st.caption("Please note: There may be a one-day delay in reflecting your latest contributions.\
                            If a project you have previously supported has a new payout address, it may not appear in this list.")

                        # Exclude donations in GG20 before finding most supported projects
                        filtered_df = final_df[final_df['Aggregate Name'] != 'GG20']

                        top_donations = filtered_df.groupby('PayoutAddress').agg({'AmountUSD': 'sum'}).reset_index()
                        
                        top_recos = get_recommendations_gg20(top_donations, address)

                        # Filter the DataFrame to include only the necessary columns
                        display_df = top_recos[['Project Name', 'Round Name', 'Donated', 'link']]
                        
                        # Prefix 'Project Name' with a checkmark if 'Donated' is 'Yes'
                        display_df['Project Name'] = display_df.apply(
                            lambda row: '✅ ' + row['Project Name'] if row['Donated'] == 'Yes' else row['Project Name'],
                            axis=1
                        )
                        
                        st.dataframe(
                            display_df,
                            hide_index=True, 
                            use_container_width=True,
                            column_order=("Project Name","Round Name", "link"),
                            column_config={
                                "link": st.column_config.LinkColumn("Explorer Link", display_text="View Project")    
                            }
                        )                    

                        my_bar.empty()

                        # Social Sharing
                        
                        tcol2.markdown("🌟 **Share Your Impact!** 🌍 Let the world know how you've contributed to the open-source ecosystem and beyond with Gitcoin Grants. Inspire others with your journey! 💫")
                        # Insert the custom URL into the HTML string using string formatting
                        html_content = f"""
                            <a href="https://twitter.com/share?ref_src=twsrc%5Etfw" class="twitter-share-button" 
                            data-text="I\'ve been a part of curating and signaling the next generation of high-impact projects through @gitcoin Grants 🫡 Join me and share your impact today! Check out my #ggwrapped Impact Dashboard " 
                            data-url="{custom_url}"
                            data-show-count="false"
                            data-size="Large" 
                            height=10
                            Tweet
                            </a>
                            <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                        """
                        
                        with tcol2:
                            components.html(html_content)


                else:
                    my_bar.empty()
                    tcol2.write("While there are contributions made from this address on Grants Stack, no contribution data found for Gitcoin Grants.")                    
            else:
                my_bar.empty()
                tcol2.write("No contribution data available for this address.")

if __name__ == "__main__":
    main()
