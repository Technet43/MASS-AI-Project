const revealed = document.querySelectorAll("[data-reveal]");
const themeToggle = document.getElementById("theme-toggle");
const THEME_KEY = "mass-ai-theme-v2";

const setTheme = (theme) => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(THEME_KEY, theme);

    if (themeToggle) {
        const nextMode = theme === "light" ? "Dark Mode" : "Light Mode";
        themeToggle.textContent = nextMode;
        themeToggle.setAttribute("aria-label", `Switch to ${nextMode.toLowerCase()}`);
    }
};

const observer = new IntersectionObserver(
    (entries) => {
        for (const entry of entries) {
            if (entry.isIntersecting) {
                entry.target.classList.add("is-visible");
                observer.unobserve(entry.target);
            }
        }
    },
    { threshold: 0.16 }
);

for (const node of revealed) {
    observer.observe(node);
}

if (themeToggle) {
    setTheme(document.documentElement.dataset.theme || "dark");

    themeToggle.addEventListener("click", () => {
        const currentTheme = document.documentElement.dataset.theme === "light" ? "light" : "dark";
        setTheme(currentTheme === "light" ? "dark" : "light");
    });
}
