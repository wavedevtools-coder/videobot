# modules/upload_manager.py
"""YouTube upload management with scheduling support."""

import logging
import os
import json
import pickle
from datetime import datetime, time, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path

from .config_loader import Config

logger = logging.getLogger('upload_manager')


class UploadQueue:
    """Manages upload queue with scheduling."""

    QUEUE_FILE = "data/upload_queue.json"
    HISTORY_FILE = "data/upload_history.json"

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.queue_file = os.path.join(
            self.config.get('project_root') or '.', self.QUEUE_FILE
        )
        self.history_file = os.path.join(
            self.config.get('project_root') or '.', self.HISTORY_FILE
        )
        self._queue: List[Dict[str, Any]] = []
        self._history: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        """Load queue and history."""
        for filepath, attr in [(self.queue_file, '_queue'), (self.history_file, '_history')]:
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        setattr(self, attr, json.load(f))
                except json.JSONDecodeError:
                    setattr(self, attr, [])
            else:
                setattr(self, attr, [])

    def _save(self):
        """Save queue and history."""
        os.makedirs(os.path.dirname(self.queue_file), exist_ok=True)
        with open(self.queue_file, 'w', encoding='utf-8') as f:
            json.dump(self._queue, f, indent=2)
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self._history, f, indent=2)

    def add(self, video_path: str, metadata: Dict[str, Any], scheduled_time: Optional[datetime] = None):
        """Add video to upload queue."""
        entry = {
            'video_path': video_path,
            'title': metadata.get('title', 'Tiny Dino Adventure'),
            'description': metadata.get('description', ''),
            'tags': metadata.get('tags', []),
            'category': 'Film & Animation',
            'privacy': 'public',
            'made_for_kids': True,
            'scheduled_time': scheduled_time.isoformat() if scheduled_time else None,
            'added_at': datetime.now().isoformat(),
            'status': 'pending',
            'attempts': 0,
        }
        self._queue.append(entry)
        self._save()
        logger.info(f"Added to upload queue: {entry['title']}")

    def get_next(self) -> Optional[Dict[str, Any]]:
        """Get next video ready for upload."""
        now = datetime.now()
        for entry in self._queue:
            if entry['status'] != 'pending':
                continue
            scheduled = entry.get('scheduled_time')
            if scheduled:
                scheduled_dt = datetime.fromisoformat(scheduled)
                if scheduled_dt > now:
                    continue
            return entry
        return None

    def mark_done(self, entry: Dict[str, Any], video_id: str = ""):
        """Mark upload as completed."""
        entry['status'] = 'uploaded'
        entry['uploaded_at'] = datetime.now().isoformat()
        entry['video_id'] = video_id
        self._history.append(entry)
        if entry in self._queue:
            self._queue.remove(entry)
        self._save()

    def mark_failed(self, entry: Dict[str, Any], error: str):
        """Mark upload as failed."""
        entry['attempts'] += 1
        entry['last_error'] = error
        if entry['attempts'] >= 3:
            entry['status'] = 'failed'
            logger.error(f"Upload failed after 3 attempts: {entry['title']}")
        self._save()

    def get_pending_count(self) -> int:
        return sum(1 for e in self._queue if e['status'] == 'pending')

    def get_today_uploaded_count(self) -> int:
        today = datetime.now().date()
        return sum(
            1 for e in self._history
            if datetime.fromisoformat(e.get('uploaded_at', '2000-01-01')).date() == today
        )

    def can_upload_today(self) -> bool:
        """Check if daily upload limit reached."""
        upload_cfg = self.config.upload
        if not upload_cfg.get('enabled', False):
            return False
        daily_limit = upload_cfg.get('videos_per_day', 1)
        return self.get_today_uploaded_count() < daily_limit


class YouTubeUploader:
    """YouTube upload using YouTube Data API v3."""

    SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
    API_SERVICE_NAME = 'youtube'
    API_VERSION = 'v3'

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.queue = UploadQueue(config)
        self.credentials = None
        self._authenticated = False

    def authenticate(self, credentials_path: str = "client_secrets.json"):
        """Authenticate with YouTube API."""
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build

            creds = None
            token_path = 'data/youtube_token.pickle'

            # Load existing token
            if os.path.exists(token_path):
                with open(token_path, 'rb') as token:
                    creds = pickle.load(token)

            # Refresh or create new credentials
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not os.path.exists(credentials_path):
                        logger.error(
                            f"YouTube client secrets not found: {credentials_path}\n"
                            "Download from Google Cloud Console > APIs & Services > Credentials"
                        )
                        return False

                    flow = InstalledAppFlow.from_client_secrets_file(
                        credentials_path, self.SCOPES)
                    creds = flow.run_local_server(port=0)

                # Save token
                os.makedirs(os.path.dirname(token_path), exist_ok=True)
                with open(token_path, 'wb') as token:
                    pickle.dump(creds, token)

            self.credentials = creds
            self._authenticated = True
            logger.info("YouTube authentication successful")
            return True

        except ImportError:
            logger.error(
                "Google API libraries not installed.\n"
                "pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client"
            )
            return False
        except Exception as e:
            logger.error(f"YouTube authentication failed: {e}")
            return False

    def upload_video(self, entry: Dict[str, Any]) -> str:
        """Upload a video to YouTube."""
        if not self._authenticated:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        try:
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload

            youtube = build(self.API_SERVICE_NAME, self.API_VERSION, credentials=self.credentials)

            body = {
                'snippet': {
                    'title': entry['title'],
                    'description': entry['description'],
                    'tags': entry.get('tags', []),
                    'categoryId': '1',  # Film & Animation
                },
                'status': {
                    'privacyStatus': entry.get('privacy', 'private'),
                    'selfDeclaredMadeForKids': entry.get('made_for_kids', True),
                },
            }

            media = MediaFileUpload(
                entry['video_path'],
                mimetype='video/mp4',
                resumable=True,
            )

            request = youtube.videos().insert(
                part='snippet,status',
                body=body,
                media_body=media,
            )

            # Execute upload with progress
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    logger.info(f"Upload progress: {int(status.progress() * 100)}%")

            video_id = response['id']
            logger.info(f"Upload complete: https://youtube.com/shorts/{video_id}")
            return video_id

        except Exception as e:
            raise RuntimeError(f"YouTube upload failed: {e}")

    def process_queue(self):
        """Process upload queue."""
        if not self.queue.can_upload_today():
            logger.info("Daily upload limit reached")
            return

        entry = self.queue.get_next()
        if not entry:
            logger.info("No pending uploads")
            return

        try:
            video_id = self.upload_video(entry)
            self.queue.mark_done(entry, video_id)
            logger.info(f"Successfully uploaded: {entry['title']}")
        except Exception as e:
            self.queue.mark_failed(entry, str(e))
            logger.error(f"Upload failed: {e}")

    def schedule_upload(self, video_path: str, metadata: Dict[str, Any]):
        """Schedule video for upload at configured time."""
        upload_times = self.config.upload.get('upload_times', ['18:00'])

        # Find next available upload time
        now = datetime.now()
        next_time = None

        for ut in upload_times:
            hour, minute = map(int, ut.split(':'))
            candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if candidate > now:
                next_time = candidate
                break

        if not next_time:
            # Use first time tomorrow
            hour, minute = map(int, upload_times[0].split(':'))
            next_time = (now + timedelta(days=1)).replace(
                hour=hour, minute=minute, second=0, microsecond=0
            )

        self.queue.add(video_path, metadata, scheduled_time=next_time)
        logger.info(f"Scheduled upload for {next_time}: {metadata.get('title')}")

    def generate_metadata(self, story: Dict[str, Any], episode: int) -> Dict[str, Any]:
        """Generate YouTube metadata from story."""
        upload_cfg = self.config.upload

        title_template = upload_cfg.get('title_template', 'Tiny Dino {episode} - {story_title}')
        desc_template = upload_cfg.get('description_template', '')
        tags = upload_cfg.get('tags', [])

        title = title_template.format(
            episode=f'#{episode:03d}',
            story_title=story.get('title', 'Adventure'),
        )

        # Build description
        scenes = story.get('scenes', [])
        scene_summaries = []
        for s in scenes:
            desc = s.get('description', '')
            if desc:
                scene_summaries.append(f"• {desc[:100]}")

        story_summary = '\n'.join(scene_summaries[:3])

        description = desc_template.format(
            episode=episode,
            story_title=story.get('title', ''),
            story_summary=story_summary,
            theme=story.get('theme', ''),
        )

        return {
            'title': title[:100],  # YouTube limit
            'description': description[:5000],
            'tags': tags[:15],  # YouTube limit
        }
