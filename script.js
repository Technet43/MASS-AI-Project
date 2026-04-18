const galleryCards = document.querySelectorAll(".gallery-card");
const lightbox = document.getElementById("lightbox");
const lightboxImage = document.getElementById("lightbox-image");
const lightboxTitle = document.getElementById("lightbox-title");
const lightboxText = document.getElementById("lightbox-text");
const lightboxClose = document.querySelector(".lightbox-close");

for (const card of galleryCards) {
    card.addEventListener("click", () => {
        lightboxImage.src = card.dataset.image || "";
        lightboxImage.alt = card.dataset.title || "Preview image";
        lightboxTitle.textContent = card.dataset.title || "";
        lightboxText.textContent = card.dataset.copy || "";
        lightbox.showModal();
    });
}

lightboxClose.addEventListener("click", () => {
    lightbox.close();
});

lightbox.addEventListener("click", (event) => {
    const bounds = lightbox.getBoundingClientRect();
    const isInside =
        event.clientX >= bounds.left &&
        event.clientX <= bounds.right &&
        event.clientY >= bounds.top &&
        event.clientY <= bounds.bottom;

    if (!isInside) {
        lightbox.close();
    }
});

document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && lightbox.open) {
        lightbox.close();
    }
});
