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
    rest::{get_ipv4_address, run_api},
};
use super::localization::{translate, Key, Lang};

// ── palette ──────────────────────────────────────────────────────────
const ACCENT: Color = Color::Rgb(51, 218, 114);

fn bold(fg: Color) -> Style { Style::default().fg(fg).add_modifier(Modifier::BOLD) }
fn dim() -> Style { Style::default().add_modifier(Modifier::DIM) }
fn accent() -> Style { Style::default().fg(ACCENT) }
fn focus_bar() -> Style { Style::default().fg(ACCENT).add_modifier(Modifier::REVERSED) }
fn centered(span: Span) -> Paragraph { Paragraph::new(span).alignment(Alignment::Center) }
fn at(area: Rect, y: u16) -> Rect { Rect { y, height: 1, ..area } }

// ── types ────────────────────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq)]
enum Row { Ai, ClsAi, Ep, Deploy }

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
    cls_ais: Vec<AIMetadata>,
    cls_active: bool,
    cls: Dropdown,
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
        let cls_ais: Vec<AIMetadata> = ais.iter().filter(|ai| ai.task == Task::Classify && ai.modality == Modality::Image).cloned().collect();
        Self {
            lang,
            row: 0, side_btn: false, open: None,
            ais, ai: Dropdown::default(),
            cls_ais, cls_active: false, cls: Dropdown::default(),
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
        if self.cls_active { v.push(Row::ClsAi); }
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
    fn can_add_cls(&self) -> bool {
        !self.cls_active
            && !self.cls_ais.is_empty()
            && self.ai.selected.is_some_and(|i| matches!(self.ais[i].task, Task::Detect | Task::Segment))
    }
    fn has_side_btn(&self) -> bool {
        match self.cur_row() {
            Row::Ai => self.can_add_cls(),
            Row::ClsAi => true,
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
            Row::ClsAi => handle_dropdown(code, app.cls_ais.len(), &mut app.cls),
            Row::Ep => handle_dropdown(code, app.eps.len(), &mut app.ep),
            Row::Deploy => None,
        };
        if let Some(changed) = changed {
            app.open = None;
            if changed {
                match which {
                    Row::Ai => load_ai_model(app),
                    Row::ClsAi => load_cls_model(app),
                    Row::Ep => { load_ai_model(app); load_cls_model(app); }
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
                    Row::Ai => { app.cls_active = true; app.side_btn = false; app.row = 1; }
                    Row::ClsAi => { app.cls_active = false; app.cls.selected = None; GlobalBQ::Second.clear(); app.side_btn = false; app.clamp(); }
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
                        Row::ClsAi => app.cls.reset_cursor(),
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
        let ep = app.ep.selected.map_or(Ep::Cpu, |i| app.eps[i]);
        let model_path = app.ais[ai_idx].get_path();
        app.status_msg = GlobalBQ::First.set_model(&model_path, ep, None)
            .err().map(|e| format!("{}: {}", app.t(Key::error_ocurred), e));
    }
}

fn load_cls_model(app: &mut Tui) {
    if let Some(cls_idx) = app.cls.selected {
        let ep = app.ep.selected.map_or(Ep::Cpu, |i| app.eps[i]);
        let model_path = app.cls_ais[cls_idx].get_path();
        app.status_msg = GlobalBQ::Second.set_model(&model_path, ep, None)
            .err().map(|e| format!("{}: {}", app.t(Key::error_ocurred), e));
    }
}

fn deploy_api(app: &mut Tui) {
    let port = 8791u16;
    match std::net::TcpListener::bind(("0.0.0.0", port)) {
        Ok(probe) => {
            drop(probe);
            tokio::spawn(async move {
                if let Err(e) = run_api(port).await {
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

fn draw_sidebar(frame: &mut Frame, app: &Tui, area: Rect) -> (Rect, Rect, Rect) {
    let block = Block::default().borders(Borders::RIGHT).border_style(dim());
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let cls_rows: u16 = if app.cls_active { 3 } else { 0 };
    let [head, ai, cls_area, ep, _gap, btn, _rest, hints] = Layout::vertical([
        Constraint::Length(3), Constraint::Length(3), Constraint::Length(cls_rows), Constraint::Length(3),
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
    if app.can_add_cls() {
        let focused = app.cur_row() == Row::Ai && app.side_btn;
        draw_side_btn(frame, ai, "+", focused);
    }

    // classification AI section
    if app.cls_active {
        draw_combo(frame, cls_area, app.t(Key::select_2nd_ai), &app.cls_ais, app.cls.selected, app.cur_row() == Row::ClsAi && !app.side_btn);
        let focused = app.cur_row() == Row::ClsAi && app.side_btn;
        draw_side_btn(frame, cls_area, "-", focused);
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

    (ai, cls_area, ep)
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

fn draw_dropdown_overlay(frame: &mut Frame, app: &Tui, which: Row, rows: (Rect, Rect, Rect)) {
    let (names, dd, title, row_area): (Vec<&str>, &Dropdown, &str, Rect) = match which {
        Row::Ai => (app.ais.iter().map(|a| a.name.as_str()).collect(), &app.ai, app.t(Key::select_ai), rows.0),
        Row::ClsAi => (app.cls_ais.iter().map(|a| a.name.as_str()).collect(), &app.cls, app.t(Key::select_2nd_ai), rows.1),
        Row::Ep => (app.eps.iter().map(|e| e.name()).collect(), &app.ep, app.t(Key::select_ep), rows.2),
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

    let items: Vec<ListItem> = names.iter().enumerate().skip(scroll).map(|(i, name)| {
        let cur = i == dd.cursor;
        let sel = dd.selected == Some(i);
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
