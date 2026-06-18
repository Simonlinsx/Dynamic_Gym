# Dynamic Gym Project Page

This directory contains a static project page for the current dynamic grasping
branch. Open `index.html` directly in a browser.

## Update Workflow

1. Put new images in `assets/images/`.
2. Put new videos in `assets/videos/`.
3. Edit `project_data.js` to add or update:
   - hero stats
   - method pipeline cards
   - experiment timeline entries
   - video gallery cards
   - current diagnosis items

The page has no build step and no external web dependencies.

## Suggested Media Commands

Create a poster frame from a new mp4:

```bash
ffmpeg -y -loglevel error -i assets/videos/new_run.mp4 -frames:v 1 -q:v 3 assets/images/new_run.jpg
```

Then add a record to `project_data.js`:

```js
{
  title: "New run title",
  tag: "vNN",
  tone: "watch",
  result: "short result",
  src: "assets/videos/new_run.mp4",
  poster: "assets/images/new_run.jpg",
  caption: "One-sentence description.",
}
```
