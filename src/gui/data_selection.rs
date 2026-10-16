use crate::api::{abstractions::Pred, formats};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

fn path_key(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

/// Append without replacing predictions or changing indexes used by active workers.
pub(super) fn append_files<T: Pred>(
    selected: &mut Vec<T>,
    paths: Vec<PathBuf>,
    new: impl Fn(PathBuf) -> T,
) -> bool {
    if paths.is_empty() {
        return false;
    }
    let mut known: HashSet<_> = selected
        .iter()
        .map(|file| path_key(file.file_path()))
        .collect();
    let old_len = selected.len();
    for path in paths {
        if known.insert(path_key(&path)) {
            selected.push(new(path));
        }
    }
    selected.len() != old_len
}

#[derive(Default)]
pub(super) struct MediaFiles {
    pub images: Vec<PathBuf>,
    pub audios: Vec<PathBuf>,
    pub videos: Vec<PathBuf>,
    pub had_errors: bool,
}

impl MediaFiles {
    pub fn from_folders(folders: Vec<PathBuf>) -> Self {
        let mut files = Self::default();
        let mut pending = folders;
        pending.reverse();
        let mut visited = HashSet::new();
        while let Some(folder) = pending.pop() {
            // Also prevents cycles through directory links and overlapping selections.
            if !visited.insert(path_key(&folder)) {
                continue;
            }
            let entries = match fs::read_dir(folder) {
                Ok(entries) => entries,
                Err(_) => {
                    files.had_errors = true;
                    continue;
                }
            };
            let mut children = Vec::new();
            for entry in entries {
                let path = match entry {
                    Ok(entry) => entry.path(),
                    Err(_) => {
                        files.had_errors = true;
                        continue;
                    }
                };
                if path.is_dir() {
                    children.push(path);
                } else if path.is_file() {
                    let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
                        continue;
                    };
                    if has_ext(ext, &formats::IMAGE_FORMATS) {
                        files.images.push(path);
                    } else if has_ext(ext, &formats::AUDIO_FORMATS) {
                        files.audios.push(path);
                    } else if has_ext(ext, &formats::VIDEO_FORMATS) {
                        files.videos.push(path);
                    }
                }
            }
            children.reverse();
            pending.extend(children);
        }
        files
    }
}

fn has_ext(ext: &str, formats: &[&str]) -> bool {
    formats
        .iter()
        .any(|format| ext.eq_ignore_ascii_case(format))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::abstractions::{PredAudio, PredImg, PredVideo};

    #[test]
    fn selections_accumulate_for_every_media_type() {
        fn check<T: Pred>(new: impl Fn(PathBuf) -> T) {
            let mut selected = Vec::new();
            assert!(append_files(
                &mut selected,
                vec!["a".into(), "b".into(), "c".into()],
                &new
            ));
            assert!(append_files(&mut selected, vec!["d".into()], &new));
            assert_eq!(selected.len(), 4);
            let paths: Vec<_> = selected.iter().map(|file| file.file_path()).collect();
            assert_eq!(
                paths,
                vec![
                    Path::new("a"),
                    Path::new("b"),
                    Path::new("c"),
                    Path::new("d")
                ]
            );
            assert!(!append_files(
                &mut selected,
                vec!["b".into(), "d".into()],
                &new
            ));
            assert!(!append_files(&mut selected, Vec::new(), &new));
            assert_eq!(selected.len(), 4);
        }
        check(PredImg::new_simple);
        check(PredAudio::new_simple);
        check(PredVideo::new_simple);
    }

    #[test]
    fn duplicates_preserve_existing_results() {
        let path = PathBuf::from("tests/assets/img.jpg");
        let mut original = PredImg::new_simple(path.clone());
        original.wasprocessed = true;
        let mut selected = vec![original];
        assert!(append_files(
            &mut selected,
            vec![
                path.canonicalize().unwrap(),
                "new.jpg".into(),
                "new.jpg".into()
            ],
            PredImg::new_simple,
        ));
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].file_path, path);
        assert!(selected[0].wasprocessed);
    }

    #[test]
    fn adding_media_preserves_navigation_playback_and_active_jobs() {
        egui::__run_test_ui(|ui| {
            let mut gui = super::super::Gui::default();
            gui.add_images(ui, vec!["a.jpg".into(), "b.jpg".into(), "c.jpg".into()]);
            gui.image_texture_n = 2;
            gui.selected_imgs[1].wasprocessed = true;
            let (_tx, _cancel) = gui.img_state.start();
            gui.add_images(ui, vec!["d.jpg".into()]);
            assert_eq!(gui.selected_imgs.len(), 4);
            assert_eq!(gui.image_texture_n, 2);
            assert!(gui.selected_imgs[1].wasprocessed);
            assert!(gui.img_state.is_active());
            assert_eq!(gui.img_state.progress_bar, 0.25);

            // Failed preview decoding must not discard the selection either.
            gui.add_audios(vec!["missing-a.wav".into(), "missing-b.wav".into()]);
            assert_eq!(gui.selected_audios.len(), 2);
            gui.audio_texture_n = 2;
            gui.audio_playhead = Some(12.0);
            gui.add_audios(vec!["missing-c.wav".into()]);
            assert_eq!(gui.selected_audios.len(), 3);
            assert_eq!(gui.audio_texture_n, 2);
            assert_eq!(gui.audio_playhead, Some(12.0));

            gui.selected_videos = vec![PredVideo::new_simple("a.mp4".into())];
            gui.video_texture_n = 1;
            gui.video_playhead_frame = Some(42);
            let (_tx, _cancel) = gui.video_state.start();
            gui.add_videos(ui, vec!["b.mp4".into()]);
            assert_eq!(gui.selected_videos.len(), 2);
            assert_eq!(gui.video_texture_n, 1);
            assert_eq!(gui.video_playhead_frame, Some(42));
            assert!(gui.video_state.is_active());
        });
    }

    #[test]
    fn folders_include_nested_media_and_tolerate_unreadable_roots() {
        // Existing fixtures cover nested folders, mixed media, and overlapping selections.
        let root = PathBuf::from("tests/assets/formats");
        let files = MediaFiles::from_folders(vec![
            root.join("image"),
            root.clone(),
            "tests/assets/perch".into(),
            root.join("missing-folder"),
        ]);
        assert!(files.had_errors);
        assert_eq!(files.images.len(), 20);
        assert_eq!(files.audios.len(), 20); // 15 audio fixtures + video.ogg + 4 Perch clips.
        assert_eq!(files.videos.len(), 33);
        let mut selected = Vec::new();
        append_files(&mut selected, files.images, PredImg::new_simple);
        append_files(
            &mut selected,
            MediaFiles::from_folders(vec![root]).images,
            PredImg::new_simple,
        );
        assert_eq!(selected.len(), 20);
    }
}
