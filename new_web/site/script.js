// Hamburger nav toggle
const navToggle = document.getElementById("nav-toggle");
const siteNav = document.getElementById("site-nav");

navToggle.addEventListener("click", () => {
    const isOpen = siteNav.classList.toggle("open");
    navToggle.classList.toggle("open", isOpen);
    navToggle.setAttribute("aria-expanded", String(isOpen));
});

siteNav.querySelectorAll("a").forEach(link => {
    link.addEventListener("click", () => {
        siteNav.classList.remove("open");
        navToggle.classList.remove("open");
        navToggle.setAttribute("aria-expanded", "false");
    });
});

// Gallery lightbox with arrow navigation
const galleryCards = Array.from(document.querySelectorAll(".gallery-card"));
const lightbox = document.getElementById("lightbox");
const lightboxImage = document.getElementById("lightbox-image");
const lightboxTitle = document.getElementById("lightbox-title");
const lightboxText = document.getElementById("lightbox-text");
const lightboxClose = document.querySelector(".lightbox-close");
const lightboxPrev = document.getElementById("lightbox-prev");
const lightboxNext = document.getElementById("lightbox-next");
const lightboxCounter = document.getElementById("lightbox-counter");

let currentIndex = 0;

function openLightbox(index) {
    currentIndex = index;
    const card = galleryCards[index];
    lightboxImage.src = card.dataset.image || "";
    lightboxImage.alt = card.dataset.title || "Preview image";
    lightboxTitle.textContent = card.dataset.title || "";
    lightboxText.textContent = card.dataset.copy || "";
    lightboxCounter.textContent = `${index + 1} / ${galleryCards.length}`;
    lightbox.showModal();
}

galleryCards.forEach((card, i) => {
    card.addEventListener("click", () => openLightbox(i));
});

lightboxPrev.addEventListener("click", () => {
    openLightbox((currentIndex - 1 + galleryCards.length) % galleryCards.length);
});

lightboxNext.addEventListener("click", () => {
    openLightbox((currentIndex + 1) % galleryCards.length);
});

lightboxClose.addEventListener("click", () => lightbox.close());

lightbox.addEventListener("click", (event) => {
    const bounds = lightbox.getBoundingClientRect();
    const isInside =
        event.clientX >= bounds.left &&
        event.clientX <= bounds.right &&
        event.clientY >= bounds.top &&
        event.clientY <= bounds.bottom;
    if (!isInside) lightbox.close();
});

document.addEventListener("keydown", (event) => {
    if (!lightbox.open) return;
    if (event.key === "Escape") lightbox.close();
    if (event.key === "ArrowLeft") openLightbox((currentIndex - 1 + galleryCards.length) % galleryCards.length);
    if (event.key === "ArrowRight") openLightbox((currentIndex + 1) % galleryCards.length);
});
