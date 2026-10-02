(() => {
  const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
  const panels = Array.from(document.querySelectorAll('[role="tabpanel"]'));

  function activate(tab) {
    tabs.forEach((t) => {
      const selected = t === tab;
      t.setAttribute("aria-selected", selected ? "true" : "false");
      t.tabIndex = selected ? 0 : -1;
    });
    panels.forEach((panel) => {
      const match = panel.id === tab.getAttribute("aria-controls");
      panel.classList.toggle("is-active", match);
      panel.hidden = !match;
    });
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activate(tab));
    tab.addEventListener("keydown", (event) => {
      const key = event.key;
      if (key !== "ArrowRight" && key !== "ArrowLeft" && key !== "Home" && key !== "End") {
        return;
      }
      event.preventDefault();
      let next = index;
      if (key === "ArrowRight") next = (index + 1) % tabs.length;
      if (key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
      if (key === "Home") next = 0;
      if (key === "End") next = tabs.length - 1;
      tabs[next].focus();
      activate(tabs[next]);
    });
  });

  if (tabs[0]) activate(tabs[0]);
})();
