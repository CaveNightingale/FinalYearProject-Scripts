
// #let color = rgb("3333b2")
#let color = black
#let slide-footer(docname: content) = context {
  let titles = query(heading.where(level: 1))
  let current_page = here().page()
  let end_page = counter(page).final().at(0)
  let current_chapter = titles.filter(t => t.location().page() <= current_page).last(default: none)
  let next_chapter = titles.filter(t => t.location().page() > current_page).first(default: none)
  let next_chapter_page = if next_chapter == none {
    end_page + 1
  } else {
    next_chapter.location().page()
  }
  let chapter_titles = titles
    .map(t => if t == current_chapter {
      let current_chapter_page = t.location().page()
      let chapter_length = next_chapter_page - current_chapter_page
      let chapter_read = current_page - current_chapter_page + 1
      let progress = block[
        #box[#rect(width: (chapter_read / chapter_length) * 10em, height: 0.5em, fill: gray, stroke: gray)]#box[#rect(width: ((chapter_length - chapter_read) / chapter_length) * 10em, height: 0.5em, stroke: gray)]
      ]
      box(width: 1fr)[#link(t.location())[#progress]]
    } else {
      box(width: 1fr)[#link(t.location())[#t.body]]
    })
    .join()
  let chapter_indicator = text(fill: gray, size: .8em)[
    #chapter_titles
  ]
  place(bottom, float: true, dx: -2cm)[
    #block(spacing: 0pt, rect(width: 100% + 4cm, fill: color, stroke: color, outset: 0pt)[
      #h(1cm)
      #chapter_indicator
      #h(1cm)
    ])
    #block(spacing: 0pt, rect(width: 100% + 4cm, fill: black, stroke: black, outset: 0pt)[
      #h(1cm)
      #text(fill: gray, size: .8em)[
        #docname
        #h(1fr)
        AY 2025-2026
      ]
      #h(1cm)
    ])
  ]
}

#let slide-header = context {
  let titles = query(heading.where(level: 1))
  let current_page = here().page()
  let end_page = counter(page).final().at(0)
  let page_indicator = text(fill: gray, size: .8em)[
    #here().page() / #end_page
  ]
  let current_chapter = titles.filter(t => t.location().page() <= current_page).last(default: none)
  if current_chapter != none {
    let second_order = query(heading.where(level: 2).after(current_chapter.location()))
      .filter(t => t.location().page() <= current_page)
      .last(default: none)
    place(top, float: true, dx: -2cm)[
      #align(horizon, block(spacing: 0pt, rect(width: 100% + 4cm, fill: color, stroke: black, outset: 0pt, height: 2em)[
        #h(1cm)
        #text(fill: gray, size: 1.6em, weight: "bold")[
          #link(current_chapter.location())[#current_chapter.body]
          #if second_order != none [ \- #link(second_order.location())[#second_order.body]]
        ]
        #h(1fr)
        #page_indicator
        #h(1cm)
      ]))
    ]
  } else {
    place(top, float: true, dx: -2cm)[
      #align(horizon, block(spacing: 0pt, rect(
        width: 100% + 4cm,
        fill: color,
        stroke: black,
        outset: 0pt,
        height: 2em,
      )[]))
    ]
  }
}
