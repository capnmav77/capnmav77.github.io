document.addEventListener("DOMContentLoaded", () => {
  const blocks = document.querySelectorAll("pre code.language-mermaid");
  if (!blocks.length) return;

  const loadMermaid = () => {
    mermaid.initialize({
      startOnLoad: false,
      theme: "base",
      themeVariables: {
        // Core Gruvbox Dark background & foreground
        background: "#282828",
        primaryColor: "#3c3836",      // Node fill (bg1)
        primaryTextColor: "#ebdbb2",  // Node text (fg)
        primaryBorderColor: "#d79921",// Node border (yellow)
        lineColor: "#a89984",         // Arrows & connectors (gray)
        textColor: "#ebdbb2",

        // Secondary & tertiary nodes (subgraphs, alternate states)
        secondaryColor: "#504945",    // bg2
        secondaryTextColor: "#ebdbb2",
        secondaryBorderColor: "#689d6a", // aqua
        tertiaryColor: "#32302f",     // bg0_s
        tertiaryTextColor: "#ebdbb2",
        tertiaryBorderColor: "#458588",  // blue

        // Notes & labels
        noteBkgColor: "#458588",      // blue
        noteTextColor: "#fbf1c7",     // fg0
        noteBorderColor: "#83a598",   // bright blue
        edgeLabelBackground: "#282828",

        // Sequence & Gantt specifics
        actorBkg: "#3c3836",
        actorBorder: "#d79921",
        actorTextColor: "#ebdbb2",
        signalColor: "#ebdbb2",
        signalTextColor: "#ebdbb2"
      }
    });

    blocks.forEach((code, index) => {
      const container = document.createElement("div");
      container.className = "mermaid";
      container.textContent = code.textContent;

      const pre = code.closest("pre");
      pre.replaceWith(container);
      container.id = `mermaid-diagram-${index}`;
    });

    mermaid.run({ querySelector: ".mermaid" });
  };

  if (window.mermaid) {
    loadMermaid();
    return;
  }

  const script = document.createElement("script");
  script.src = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js";
  script.onload = loadMermaid;
  document.head.appendChild(script);
});
