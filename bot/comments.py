"""
Module for collecting comments from youtube videoes.
"""

import logging
from pandas import DataFrame
import googleapiclient.discovery

logging.basicConfig(level=logging.INFO)


def comments_scraper(video_id, maxResults=200, DEVELOPER_KEY=""):
    """
    Get comment from video based on its id.
    """

    try:

        api_service_name = "youtube"
        api_version = "v3"

        youtube = googleapiclient.discovery.build(
            api_service_name,
            api_version,
            developerKey = DEVELOPER_KEY
        )

        request = youtube.commentThreads().list(
            part = "snippet",
            videoId = video_id,
            maxResults = maxResults
        )

        response = request.execute()
        comments = []

        for item in response['items']:
            comment = item["snippet"]['topLevelComment']['snippet']
            comments.append([
                comment['authorDisplayName'],
                comment['publishedAt'],
                comment['updatedAt'],
                comment['likeCount'],
                comment['textDisplay']
            ])

        df = DataFrame(comments, columns = ['author', 'published_at',
                                            'updated_at', 'like_count', 'text'])
        df = df.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
        df.to_csv("1d.csv")

    except Exception as e:
        logging.inf("Oops, unable to scrape comments! %s", str(e))


if __name__ == "__main__":
    developer_key = None
    comments_scraper(video_id="lTqxp6nl5O4", DEVELOPER_KEY=developer_key)
