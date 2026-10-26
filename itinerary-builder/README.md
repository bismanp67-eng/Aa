# Itinerary Builder

A simple, no-install itinerary builder for travel agents. Open `index.html` in any modern browser. No server, build step, or account is needed.

## Features

- **Drag and drop**: drag activity blocks (Flight, Hotel, Transfer, Tour, Meal, Free time, Note) onto any day. Drag cards to reorder them or move them between days.
- **Quick add**: click a block to add it to the first day, or use "+ Add item" on any day.
- **Easy edit**: click a card to edit its title, time, location, price, description, image and day.
- **Images**: paste an image URL or upload a photo. Uploads are resized to 900px wide to keep the file small.
- **Mobile-friendly**: use the ↑ ↓ buttons on each card to reorder without dragging.
- **Sort by time** per day, duplicate items, and running totals in the trip currency.
- **Autosave** in the browser. Nothing leaves your computer.
- **Export / Import JSON** to back up a trip or share it with a colleague.
- **Print / Save PDF** using the browser's print dialog. The printout hides the editing controls.

## Notes

- Data is stored in the browser's local storage for this device and browser. Export JSON regularly to keep a backup.
- Drag and drop uses standard HTML5 drag events, so it works with a mouse on desktop. On touch devices, use the ↑ ↓ buttons or the Edit dialog instead.
- Uploaded photos are stored inside the trip data. Keep the number of photos reasonable (browser storage is typically about 5 MB).
