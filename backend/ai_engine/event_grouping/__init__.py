import exifread
from datetime import datetime, timedelta
from gallery.models import Photo, Event

WINDOW_HOURS = 4  # photos within 4 hours → same event


def get_exif_datetime(image_path):
    """Extract datetime from EXIF data. Returns datetime or None."""
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, stop_tag='EXIF DateTimeOriginal', details=False)
        
        tag = tags.get('EXIF DateTimeOriginal') or tags.get('Image DateTime')
        if tag:
            return datetime.strptime(str(tag), '%Y:%m:%d %H:%M:%S')
    except Exception:
        pass
    return None


def group_photos_into_events():
    """
    Full re-grouping of all photos into events based on EXIF timestamps.
    Photos with no EXIF timestamp get assigned to a 'No Date' event.
    Returns summary dict.
    """
    # Wipe existing events
    Event.objects.all().delete()

    # Reset all photo event assignments
    Photo.objects.all().update(event=None)

    # Load all photos with their timestamps
    photos = list(Photo.objects.all())

    dated   = []  # (photo, datetime)
    no_date = []  # photos with no EXIF

    for photo in photos:
        dt = get_exif_datetime(photo.image.path)
        if dt:
            dated.append((photo, dt))
        else:
            # Fall back to upload time
            dated.append((photo, photo.uploaded_at.replace(tzinfo=None)))

    # Sort by datetime
    dated.sort(key=lambda x: x[1])

    # Cluster into groups — sliding 4-hour window
    groups = []
    current_group = []

    for photo, dt in dated:
        if not current_group:
            current_group.append((photo, dt))
        else:
            last_dt = current_group[-1][1]
            if dt - last_dt <= timedelta(hours=WINDOW_HOURS):
                current_group.append((photo, dt))
            else:
                groups.append(current_group)
                current_group = [(photo, dt)]

    if current_group:
        groups.append(current_group)

    # Create Event records
    events_created = 0
    for group in groups:
        start_dt = group[0][1]
        end_dt   = group[-1][1]
        count    = len(group)

        # Name: "Dec 12, 2024 · Evening" style
        name = format_event_name(start_dt, count)

        event = Event.objects.create(
            name       = name,
            start_time = start_dt,
            end_time   = end_dt,
        )

        for photo, _ in group:
            photo.event = event
            photo.save(update_fields=['event'])

        events_created += 1

    return {
        'status':         'ok',
        'events_created': events_created,
        'photos_grouped': len(dated),
        'no_date':        len(no_date),
    }


def assign_event_to_photo(photo):
    """
    Called on single photo upload — finds which existing event
    this photo belongs to, or creates a new one.
    """
    dt = get_exif_datetime(photo.image.path)
    if not dt:
        dt = photo.uploaded_at.replace(tzinfo=None)

    # Find an existing event within WINDOW_HOURS
    window = timedelta(hours=WINDOW_HOURS)
    matched_event = None

    for event in Event.objects.all():
        event_start = event.start_time
        event_end   = event.end_time

        # If photo falls within or adjacent to this event's window
        if (event_start - window) <= dt <= (event_end + window):
            matched_event = event
            # Expand event window if needed
            if dt < event_start:
                event.start_time = dt
            if dt > event_end:
                event.end_time = dt
            event.save()
            break

    if not matched_event:
        name = format_event_name(dt, 1)
        matched_event = Event.objects.create(
            name       = name,
            start_time = dt,
            end_time   = dt,
        )

    photo.event = matched_event
    photo.save(update_fields=['event'])
    return matched_event


def format_event_name(dt, count):
    """Generate a human-readable event name from a datetime."""
    hour = dt.hour
    if   5  <= hour < 12: period = "Morning"
    elif 12 <= hour < 17: period = "Afternoon"
    elif 17 <= hour < 21: period = "Evening"
    else:                  period = "Night"

    date_str = dt.strftime("%b %d, %Y")
    return f"{date_str} · {period}"