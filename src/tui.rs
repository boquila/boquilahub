use std::time::Duration;

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph};
use ratatui::Frame;

use super::api::{
    bq::{AIMetadata, BQModel, Ep, GlobalBQ, Modality},
    models::Task,
    rest::{get_ipv4_address, Rest},
};
use super::localization::{translate, Key, Lang};

// ── palette ──────────────────────────────────────────────────────────
const ACCENT: Color = Color::Rgb(51, 218, 114);
const EXTRA_AI_SLOTS: [GlobalBQ; 2] = [GlobalBQ::Second, GlobalBQ::Third];

fn bold(fg: Color) -> Style { Style::default().fg(fg).add_modifier(Modifier::BOLD) }
fn dim() -> Style { Style::default().add_modifier(Modifier::DIM) }
fn accent() -> Style { Style::default().fg(ACCENT) }
fn focus_bar() -> Style { Style::default().fg(ACCENT).add_modifier(Modifier::REVERSED) }
fn centered(span: Span) -> Paragraph { Paragraph::new(span).alignment(Alignment::Center) }
fn at(area: Rect, y: u16) -> Rect { Rect { y, height: 1, ..area } }

// ── types ────────────────────────────────────────────────────────────
#[derive(Clone, Copy, Debug, PartialEq)]
enum Row { Ai, ExtraAi(usize), Ep, Deploy }

#[derive(Default)]
struct Dropdown {
    selected: Option<usize>,
    cursor: usize,
}

impl Dropdown {
    fn reset_cursor(&mut self) {
        self.cursor = self.selected.unwrap_or(0);
    }
}

pub struct Tui {
    lang: Lang,
    row: usize,
    side_btn: bool,     // true = focus is on the +/- button, not the combo
    open: Option<Row>,  // which dropdown, if any, is currently open
    ais: Vec<AIMetadata>,
    ai: Dropdown,
    extra_ais: Vec<Dropdown>,
    eps: Vec<Ep>,
    ep: Dropdown,
    api_deployed: bool,
    host_url: Option<String>,
    status_msg: Option<String>,
}

impl Tui {
    pub fn run(lang: Lang) -> std::io::Result<()> {
        let mut app = Tui::new(lang);
        ratatui::run(|terminal| loop {
            terminal.draw(|f| draw(f, &app))?;
            if event::poll(Duration::from_millis(50))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press && handle_input(&mut app, key.code, key.modifiers) {
                        break Ok(());
                    }
                }
            }
        })
    }
    
    fn new(lang: Lang) -> Self {
        let ais = BQModel::get_list();
        Self {
            lang,
            row: 0, side_btn: false, open: None,
            ais, ai: Dropdown::default(),
            extra_ais: Vec::new(),
            eps: Ep::locals(), ep: Dropdown::default(),
            api_deployed: false,
            host_url: None,
            status_msg: None,
        }
    }
    fn t(&self, key: Key) -> &'static str {
        translate(key, &self.lang)
    }
    fn rows(&self) -> Vec<Row> {
        let mut v = vec![Row::Ai];
        v.extend((0..self.extra_ais.len()).map(Row::ExtraAi));
        v.push(Row::Ep);
        if self.can_deploy() || self.api_deployed { v.push(Row::Deploy); }
        v
    }
    fn cur_row(&self) -> Row {
        let rows = self.rows();
        rows[self.row.min(rows.len() - 1)]
    }
    fn can_deploy(&self) -> bool {
        self.ai.selected.is_some() && self.ep.selected.is_some()
    }
    fn extra_ai_choices(&self, slot: usize) -> Vec<usize> {
        self.ais.iter().enumerate().filter_map(|(index, ai)| {
            let available = ai.modality == Modality::Image
                && matches!(ai.task, Task::Classify | Task::Embed)
                && !self.extra_ais.iter().enumerate().any(|(other, extra)| {
                    other != slot && extra.selected.is_some_and(|i| self.ais[i].task == ai.task)
                });
            available.then_some(index)
        }).collect()
    }
    fn can_add_extra_ai(&self) -> bool {
        self.ai.selected.is_some_and(|i| self.ais[i].can_add_cls())
            && self.extra_ais.len() < EXTRA_AI_SLOTS.len()
            && self.extra_ais.iter().all(|slot| slot.selected.is_some())
            && !self.extra_ai_choices(self.extra_ais.len()).is_empty()
    }
    fn clear_extra_ais(&mut self) {
        self.extra_ais.clear();
        for slot in EXTRA_AI_SLOTS { slot.clear(); }
        self.clamp();
    }
    fn remove_extra_ai(&mut self, index: usize) {
        self.extra_ais.remove(index);
        EXTRA_AI_SLOTS[index].clear();
        if index < self.extra_ais.len() {
            GlobalBQ::Second.move_model_from(GlobalBQ::Third);
        }
        self.side_btn = false;
        self.clamp();
    }
    fn has_side_btn(&self) -> bool {
        match self.cur_row() {
            Row::Ai => self.can_add_extra_ai(),
            Row::ExtraAi(_) => true,
            _ => false,
        }
    }
    fn clamp(&mut self) {
        let len = self.rows().len();
        if self.row >= len { self.row = len - 1; }
        if !self.has_side_btn() { self.side_btn = false; }
    }
}

// ── input ────────────────────────────────────────────────────────────
fn handle_input(app: &mut Tui, code: KeyCode, mods: KeyModifiers) -> bool {
    if code == KeyCode::Char('c') && mods.contains(KeyModifiers::CONTROL) { return true; }
    if matches!(code, KeyCode::Char('q') | KeyCode::Esc) && app.open.is_none() { return true; }

    if let Some(which) = app.open {
        let changed = match which {
            Row::Ai => handle_dropdown(code, app.ais.len(), &mut app.ai),
            Row::ExtraAi(index) => {
                let choices = app.extra_ai_choices(index);
                // The selection indexes the full catalog; the cursor indexes the filtered menu.
                let slot = &mut app.extra_ais[index];
                let mut menu = Dropdown {
                    selected: choices.iter().position(|&i| Some(i) == slot.selected),
                    cursor: slot.cursor,
                };
                let changed = handle_dropdown(code, choices.len(), &mut menu);
                slot.cursor = menu.cursor;
                if changed == Some(true) {
                    slot.selected = menu.selected.map(|i| choices[i]);
                }
                changed
            }
            Row::Ep => handle_dropdown(code, app.eps.len(), &mut app.ep),
            Row::Deploy => None,
        };
        if let Some(changed) = changed {
            app.open = None;
            if changed {
                app.status_msg = None;
                match which {
                    Row::Ai => load_ai_model(app),
                    Row::ExtraAi(index) => load_extra_ai_model(app, index),
                    Row::Ep => {
                        load_ai_model(app);
                        if app.ai.selected.is_some() {
                            for index in 0..app.extra_ais.len() {
                                load_extra_ai_model(app, index);
                            }
                        }
                    }
                    Row::Deploy => {}
                }
            }
        }
        return false;
    }

    let row_len = app.rows().len();
    match code {
        KeyCode::Tab | KeyCode::Down => { app.row = (app.row + 1) % row_len; app.side_btn = false; }
        KeyCode::BackTab | KeyCode::Up => { app.row = (app.row + row_len - 1) % row_len; app.side_btn = false; }
        KeyCode::Right => { if app.has_side_btn() { app.side_btn = true; } }
        KeyCode::Left  => { app.side_btn = false; }
        KeyCode::Enter => {
            if app.side_btn {
                match app.cur_row() {
                    Row::Ai if app.can_add_extra_ai() => {
                        app.extra_ais.push(Dropdown::default());
                        app.side_btn = false;
                        app.row = app.extra_ais.len();
                    }
                    Row::ExtraAi(index) => app.remove_extra_ai(index),
                    _ => {}
                }
            } else {
                let row = app.cur_row();
                if row == Row::Deploy {
                    if !app.api_deployed && app.can_deploy() {
                        deploy_api(app);
                    }
                } else {
                    match row {
                        Row::Ai => app.ai.reset_cursor(),
                        Row::ExtraAi(index) => {
                            app.extra_ais[index].cursor = app.extra_ai_choices(index).iter()
                                .position(|&i| Some(i) == app.extra_ais[index].selected).unwrap_or(0);
                        }
                        Row::Ep => app.ep.reset_cursor(),
                        Row::Deploy => unreachable!(),
                    }
                    app.open = Some(row);
                }
            }
        }
        _ => {}
    }
    false
}

fn handle_dropdown(code: KeyCode, len: usize, dd: &mut Dropdown) -> Option<bool> {
    match code {
        KeyCode::Up => { dd.cursor = dd.cursor.saturating_sub(1); None }
        KeyCode::Down => { dd.cursor = (dd.cursor + 1).min(len.saturating_sub(1)); None }
        KeyCode::Enter => {
            if len == 0 { return Some(false); }
            let changed = dd.selected != Some(dd.cursor);
            dd.selected = Some(dd.cursor);
            Some(changed)
        }
        KeyCode::Esc => Some(false),
        _ => None,
    }
}

fn load_ai_model(app: &mut Tui) {
    if let Some(ai_idx) = app.ai.selected {
        if !app.ais[ai_idx].can_add_cls() {
            app.clear_extra_ais();
        }
        let ep = app.ep.selected.map_or(Ep::Cpu, |i| app.eps[i]);
        let model_path = app.ais[ai_idx].get_path();
        GlobalBQ::First.clear();
        if let Err(e) = GlobalBQ::First.set_model(&model_path, ep, None) {
            let error = format!("{}: {}", app.t(Key::error_ocurred), e);
            app.status_msg.get_or_insert(error);
            app.ai.selected = None;
            app.clear_extra_ais();
        }
    }
}

fn load_extra_ai_model(app: &mut Tui, index: usize) {
    if let Some(selected) = app.extra_ais[index].selected {
        let ep = app.ep.selected.map_or(Ep::Cpu, |i| app.eps[i]);
        let target = EXTRA_AI_SLOTS[index];
        target.clear();
        if let Err(e) = target.set_model(&app.ais[selected].get_path(), ep, None) {
            let error = format!("{}: {}", app.t(Key::error_ocurred), e);
            app.status_msg.get_or_insert(error);
            app.extra_ais[index].selected = None;
        }
    }
}

fn deploy_api(app: &mut Tui) {
    let port = 8791u16;
    match std::net::TcpListener::bind(("0.0.0.0", port)) {
        Ok(probe) => {
            drop(probe);
            tokio::spawn(async move {
                if let Err(e) = Rest::deploy(port).await {
                    eprintln!("API error: {}", e);
                }
            });
            app.host_url = get_ipv4_address().map(|ip| format!("http://{}:{}", ip, port));
            app.api_deployed = true;
        }
        Err(e) => app.status_msg = Some(format!("{}: {}", app.t(Key::error_ocurred), e)),
    }
}

// ── drawing ──────────────────────────────────────────────────────────
fn draw(frame: &mut Frame, app: &Tui) {
    let [title, body, status] = Layout::vertical([
        Constraint::Length(1), Constraint::Min(0), Constraint::Length(1),
    ]).areas(frame.area());
    let [sidebar, central] = Layout::horizontal([Constraint::Length(28), Constraint::Min(0)]).areas(body);

    frame.render_widget(Paragraph::new(Span::styled(" BoquilaHUB ", bold(ACCENT))), title);
    let combo_rows = draw_sidebar(frame, app, sidebar);
    draw_central(frame, app, central);
    draw_status_bar(frame, app, status);

    if let Some(which) = app.open {
        draw_dropdown_overlay(frame, app, which, combo_rows);
    }
}

fn draw_sidebar(frame: &mut Frame, app: &Tui, area: Rect) -> (Rect, Rect, Rect, Rect) {
    let block = Block::default().borders(Borders::RIGHT).border_style(dim());
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let second_rows: u16 = if !app.extra_ais.is_empty() { 3 } else { 0 };
    let third_rows: u16 = if app.extra_ais.len() > 1 { 3 } else { 0 };
    let [head, ai, second_area, third_area, ep, _gap, btn, _rest, hints] = Layout::vertical([
        Constraint::Length(3), Constraint::Length(3), Constraint::Length(second_rows), Constraint::Length(third_rows), Constraint::Length(3),
        Constraint::Length(2), Constraint::Length(3), Constraint::Min(0), Constraint::Length(2),
    ]).areas(inner);

    // heading
    frame.render_widget(
        Paragraph::new(Line::from(vec![Span::styled("◈ ", accent()), Span::styled(app.t(Key::setup), bold(Color::Reset))]))
            .alignment(Alignment::Center),
        at(head, head.y + 1),
    );

    draw_combo(frame, ai, app.t(Key::select_ai), &app.ais, app.ai.selected, app.cur_row() == Row::Ai && !app.side_btn);

    // [+] button to the right of the Model combo
    if app.can_add_extra_ai() {
        let focused = app.cur_row() == Row::Ai && app.side_btn;
        draw_side_btn(frame, ai, "+", focused);
    }

    for (index, slot) in app.extra_ais.iter().enumerate() {
        let area = [second_area, third_area][index];
        let label = app.t(if index == 0 { Key::select_2nd_ai } else { Key::select_3rd_ai });
        let focused = app.cur_row() == Row::ExtraAi(index);
        draw_combo(frame, area, label, &app.ais, slot.selected, focused && !app.side_btn);
        draw_side_btn(frame, area, "-", focused && app.side_btn);
    }

    draw_combo(frame, ep, app.t(Key::select_ep), &app.eps, app.ep.selected, app.cur_row() == Row::Ep);

    // deploy button — only visible when both AI and processor are chosen
    if app.can_deploy() || app.api_deployed {
        let focused = app.cur_row() == Row::Deploy;
        let label_text = if app.api_deployed { app.t(Key::api_live) } else { app.t(Key::deploy_api) };
        let label = &format!(" {}", label_text);
        let style = match (focused, app.api_deployed) {
            (true, _)      => focus_bar(),
            (false, true)  => accent(),
            (false, false) => dim(),
        };
        frame.render_widget(
            Paragraph::new(Span::styled(label, style)).alignment(Alignment::Center),
            Rect { x: btn.x + 2, y: btn.y + 1, width: btn.width.saturating_sub(4), height: 1 },
        );
    }

    // hints
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("↑↓", accent()), Span::styled(format!(" {}  ", app.t(Key::nav_hint)), dim()),
            Span::styled("⏎", accent()),  Span::styled(format!(" {}", app.t(Key::select_hint)), dim()),
        ])).alignment(Alignment::Center),
        at(hints, hints.y + 1),
    );

    (ai, second_area, third_area, ep)
}

fn draw_side_btn(frame: &mut Frame, row_area: Rect, symbol: &str, focused: bool) {
    let style = if focused { focus_bar() } else { dim() };
    let bx = row_area.x + row_area.width.saturating_sub(3);
    frame.render_widget(
        Paragraph::new(Span::styled(format!("[{symbol}]"), style)),
        Rect { x: bx, y: row_area.y + 1, width: 3, height: 1 },
    );
}

fn draw_combo(frame: &mut Frame, area: Rect, label: &str, options: &[impl AsRef<str>], selected: Option<usize>, focused: bool) {
    if area.height < 2 { return; }

    let lbl = Rect { x: area.x + 2, y: area.y, width: area.width.saturating_sub(2), height: 1 };
    let combo = Rect { x: area.x + 2, y: area.y + 1, width: area.width.saturating_sub(4), height: 1 };

    frame.render_widget(Paragraph::new(Span::styled(label, if focused { accent() } else { dim() })), lbl);

    let text = selected.map_or("—".into(), |i| options[i].as_ref().to_owned());
    let style = if focused { focus_bar() } else { dim() };

    let w = combo.width as usize;
    let chev = " ▾";
    let max = w.saturating_sub(chev.chars().count());
    let trunc: String = text.chars().take(max).collect();
    let pad = max.saturating_sub(trunc.chars().count());

    frame.render_widget(Paragraph::new(Line::from(vec![
        Span::styled(format!(" {trunc}{}", " ".repeat(pad)), style),
        Span::styled(chev, style),
    ])), combo);
}

fn draw_central(frame: &mut Frame, app: &Tui, area: Rect) {
    let cy = area.y + area.height / 2;
    if app.api_deployed {
        frame.render_widget(centered(Span::styled("●", bold(ACCENT))), at(area, cy.saturating_sub(1)));
        frame.render_widget(centered(Span::styled(app.t(Key::deployed_api), dim())), at(area, cy + 1));
        if let Some(url) = &app.host_url {
            let focused = app.cur_row() == Row::Deploy;
            if focused {
                frame.render_widget(centered(Span::styled(url.as_str(), bold(Color::Reset))), at(area, cy + 3));
            } else {
                frame.render_widget(centered(Span::styled(app.t(Key::focus_deploy_to_reveal_ip), dim())), at(area, cy + 3));
            }
        }
    } else {
        if cy >= 2 {
            frame.render_widget(centered(Span::styled("◇", dim())), at(area, cy - 2));
        }
        frame.render_widget(centered(Span::styled(app.t(Key::no_api_running), dim())), at(area, cy));
        frame.render_widget(centered(Span::styled(app.t(Key::select_model_and_deploy), dim())), at(area, cy + 1));
    }
}

fn draw_status_bar(frame: &mut Frame, app: &Tui, area: Rect) {
    if let Some(msg) = &app.status_msg {
        frame.render_widget(Paragraph::new(Span::styled(format!(" {msg} "), Style::default())), area);
    }
}

fn draw_dropdown_overlay(frame: &mut Frame, app: &Tui, which: Row, rows: (Rect, Rect, Rect, Rect)) {
    let (names, dd, title, row_area): (Vec<&str>, &Dropdown, &str, Rect) = match which {
        Row::Ai => (app.ais.iter().map(|a| a.name.as_str()).collect(), &app.ai, app.t(Key::select_ai), rows.0),
        Row::ExtraAi(index) => (
            app.extra_ai_choices(index).iter().map(|&i| app.ais[i].name.as_str()).collect(),
            &app.extra_ais[index],
            app.t(if index == 0 { Key::select_2nd_ai } else { Key::select_3rd_ai }),
            if index == 0 { rows.1 } else { rows.2 },
        ),
        Row::Ep => (app.eps.iter().map(|e| e.name()).collect(), &app.ep, app.t(Key::select_ep), rows.3),
        Row::Deploy => return,
    };

    let y = row_area.y + 1; // combo line, matches draw_combo's `area.y + 1`
    let max_h = frame.area().height.saturating_sub(y);
    if max_h < 3 { return; }
    let popup_h = (names.len() as u16 + 2).min(max_h);
    let visible = popup_h.saturating_sub(2) as usize; // inner rows (minus border)
    let scroll = if dd.cursor >= visible { dd.cursor - visible + 1 } else { 0 };
    let popup = Rect { x: row_area.x + 2, y, width: 24, height: popup_h };
    frame.render_widget(Clear, popup);

    let selected = if let Row::ExtraAi(index) = which {
        app.extra_ai_choices(index).iter().position(|&i| Some(i) == dd.selected)
    } else {
        dd.selected
    };
    let items: Vec<ListItem> = names.iter().enumerate().skip(scroll).map(|(i, name)| {
        let cur = i == dd.cursor;
        let sel = selected == Some(i);
        let base = if cur { focus_bar() } else { Style::default() };
        ListItem::new(Line::from(vec![
            Span::styled(if sel { "● " } else { "  " }, if sel { base.fg(ACCENT) } else { base }),
            Span::styled(*name, base),
        ]))
    }).collect();

    frame.render_widget(List::new(items).block(
        Block::bordered()
            .title(Line::from(vec![
                Span::styled(" ", accent()),
                Span::styled(title, bold(Color::Reset)),
                Span::styled(" ", accent()),
            ]))
            .border_style(accent()),
    ), popup);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{backend::TestBackend, Terminal};

    fn app() -> Tui {
        let ais = [
            ("Detector", Task::Detect, Modality::Image),
            ("Classifier", Task::Classify, Modality::Image),
            ("Embedding", Task::Embed, Modality::Image),
            ("Other classifier", Task::Classify, Modality::Image),
            ("Audio classifier", Task::Classify, Modality::Audio),
        ].into_iter().map(|(name, task, modality)| AIMetadata {
            name: name.into(), task, modality,
            architecture: String::new(), post_processing: vec![], classes: vec![], audio_config: None,
        }).collect();
        Tui {
            lang: Lang::EN, row: 0, side_btn: false, open: None,
            ais, ai: Dropdown { selected: Some(0), cursor: 0 }, extra_ais: vec![],
            eps: vec![Ep::Cpu], ep: Dropdown::default(),
            api_deployed: false, host_url: None, status_msg: None,
        }
    }

    fn press(app: &mut Tui, code: KeyCode) {
        assert!(!handle_input(app, code, KeyModifiers::NONE));
    }

    fn add_slot(app: &mut Tui) {
        app.row = 0;
        press(app, KeyCode::Right);
        press(app, KeyCode::Enter);
    }

    fn screen(app: &Tui) -> String {
        let mut terminal = Terminal::new(TestBackend::new(80, 30)).unwrap();
        terminal.draw(|frame| draw(frame, app)).unwrap();
        let buffer = terminal.backend().buffer();
        (0..30).map(|y| {
            (0..80).map(|x| buffer[(x, y)].symbol()).collect::<String>()
        }).collect::<Vec<_>>().join("\n")
    }

    #[test]
    fn extra_selectors_appear_only_after_adding_them() {
        let mut app = app();
        assert_eq!(app.rows(), vec![Row::Ai, Row::Ep]);
        let initial = screen(&app);
        assert!(!initial.contains(app.t(Key::select_2nd_ai)));
        assert!(!initial.contains("Embedding"));
        add_slot(&mut app);
        assert_eq!(app.cur_row(), Row::ExtraAi(0));
        assert!(screen(&app).contains(app.t(Key::select_2nd_ai)));
        assert!(!screen(&app).contains(app.t(Key::select_3rd_ai)));
        assert!(!app.can_add_extra_ai());
        assert_eq!(app.extra_ai_choices(0), vec![1, 2, 3]);
        press(&mut app, KeyCode::Enter);
        let menu = screen(&app);
        assert!(menu.contains("Classifier"));
        assert!(menu.contains("Embedding"));
        assert!(!menu.contains("Audio classifier"));
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.extra_ais[0].selected, None);
    }

    #[test]
    fn either_type_can_come_first_and_third_slot_offers_only_the_other_type() {
        for (second, third, remaining) in [(1, 2, vec![2]), (2, 3, vec![1, 3])] {
            let mut app = app();
            add_slot(&mut app);
            // Supply a successful selection without loading an ONNX model in this UI test.
            app.extra_ais[0].selected = Some(second);
            assert!(app.can_add_extra_ai());
            add_slot(&mut app);
            assert_eq!(app.cur_row(), Row::ExtraAi(1));
            assert!(screen(&app).contains(app.t(Key::select_3rd_ai)));
            assert_eq!(app.extra_ai_choices(1), remaining);
            app.extra_ais[1].selected = Some(third);
            assert!(!app.can_add_extra_ai());
            assert!(app.extra_ai_choices(0).iter().all(|&i| app.ais[i].task != app.ais[third].task));

            // Reopening uses the filtered index; confirming the same model must not reload it.
            press(&mut app, KeyCode::Enter);
            assert_eq!(app.extra_ais[1].cursor, remaining.iter().position(|&i| i == third).unwrap());
            press(&mut app, KeyCode::Enter);
            assert_eq!(app.extra_ais[1].selected, Some(third));
            assert!(app.status_msg.is_none());
        }
    }

    #[test]
    fn removing_second_slot_promotes_third_and_restores_available_type() {
        let mut app = app();
        add_slot(&mut app);
        app.extra_ais[0].selected = Some(2);
        add_slot(&mut app);
        app.extra_ais[1].selected = Some(1);
        app.row = 1;
        press(&mut app, KeyCode::Right);
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.extra_ais.len(), 1);
        assert_eq!(app.extra_ais[0].selected, Some(1));
        assert_eq!(app.extra_ai_choices(1), vec![2]);
        assert!(app.can_add_extra_ai());
        assert!(!screen(&app).contains(app.t(Key::select_3rd_ai)));
        press(&mut app, KeyCode::Right);
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.rows(), vec![Row::Ai, Row::Ep]);
        assert!(!app.side_btn);
    }

    #[test]
    fn add_button_requires_a_compatible_primary_and_an_available_type() {
        let mut app = app();
        app.ai.selected = None;
        assert!(!app.can_add_extra_ai());
        app.ai.selected = Some(1);
        assert!(!app.can_add_extra_ai());
        app.ai.selected = Some(0);
        app.ais.retain(|ai| ai.task != Task::Classify);
        assert!(app.can_add_extra_ai());
        add_slot(&mut app);
        app.extra_ais[0].selected = Some(1);
        assert!(!app.can_add_extra_ai());

        let mut empty = Dropdown::default();
        assert_eq!(handle_dropdown(KeyCode::Enter, 0, &mut empty), Some(false));
        assert_eq!(empty.selected, None);
    }
}
