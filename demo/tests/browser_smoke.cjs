// Optional real-data acceptance check. Requires a running viewer and Playwright.
const assert = require("node:assert/strict");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const base = process.argv[2] || "http://127.0.0.1:8765";

async function main() {
  const browser = await chromium.launch({
    headless: true,
    args: ["--autoplay-policy=no-user-gesture-required", "--disable-gpu"],
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  try {
    await page.goto(base);
    await page.waitForSelector("button.utterance");
    const runId = await page.locator("#run-select").inputValue();
    const path = "/api/runs/" + runId + "/episodes";
    const listed = await (await page.request.get(base + path)).json();
    assert.equal(await page.locator(".episode-item").count(), listed.episodes.length);
    const current = await page.locator('.episode-item[aria-current="true"]').getAttribute("data-episode");
    const detail = await (await page.request.get(base + path + "/" + current)).json();
    assert.equal(await page.locator(".utterance").count(), detail.playback.entries.length);
    assert.equal(await page.locator(".evaluation-report").count(), detail.evaluations.length);
    await page.waitForFunction(() => document.querySelector("audio")?.readyState >= 1);
    await page.locator("audio").evaluate((audio) => audio.play());
    await page.waitForFunction(() => document.querySelector("audio").currentTime > 0.1);
    await page.locator("audio").evaluate((audio) => audio.pause());
    const targetIndex = detail.playback.entries.findIndex((entry) => entry.start > 1);
    assert.ok(targetIndex >= 0, "Need two timed utterances for the acceptance check");
    await page.locator('.utterance[data-index="' + targetIndex + '"]').click();
    const audioState = await page.locator("audio").evaluate((audio) => ({
      time: audio.currentTime, paused: audio.paused,
    }));
    assert.ok(Math.abs(audioState.time - detail.playback.entries[targetIndex].start) < 0.1);
    assert.equal(audioState.paused, true);
    assert.ok(await page.locator('.utterance[data-index="' + targetIndex + '"]').evaluate(
      (node) => node.classList.contains("is-playing"),
    ));

    for (const title of ["Agent profile", "Environment profile", "Relationship profile"]) {
      const toggle = page.locator(".profiles > details > summary").filter({ hasText: title }).first();
      await toggle.click();
      assert.ok(await toggle.evaluate((node) => node.parentElement.open));
      await toggle.click();
    }
    await page.evaluate(() => window.scrollTo(0, 0));
    if (process.env.DEMO_SCREENSHOT_PREFIX) {
      await page.screenshot({ path: process.env.DEMO_SCREENSHOT_PREFIX + "-desktop.png" });
    }
    const simulationReport = page.locator(".conversation-column .report");
    await simulationReport.locator("summary").click();
    await simulationReport.locator(".markdown h1").waitFor();
    await simulationReport.locator(".source-toggle").click();
    assert.ok(await simulationReport.locator(".markdown-source").isVisible());
    for (const report of await page.locator(".evaluation-column .report").all()) {
      await report.locator("summary").click();
      await report.locator(".markdown table").waitFor();
    }
    assert.equal(await page.evaluate(() => window.executed), undefined);

    await page.locator("audio").evaluate((audio) => {
      window.previousAudio = audio;
      return audio.play();
    });
    const next = listed.episodes.find((episode) => episode.id !== current && episode.status === "completed");
    assert.ok(next);
    await page.locator('[data-episode="' + next.id + '"]').click();
    await page.waitForFunction((id) => document.querySelector("h1")?.textContent === id.replace("episode_", "Episode "), next.id);
    assert.ok(await page.evaluate(() => window.previousAudio.paused && !window.previousAudio.hasAttribute("src")));

    // Two selections without waiting exercise cancellation of the first response.
    const alternatives = listed.episodes.filter((item) => item.status === "completed").slice(0, 3);
    await page.evaluate((ids) => {
      document.querySelector('[data-episode="' + ids[1] + '"]').click();
      document.querySelector('[data-episode="' + ids[2] + '"]').click();
    }, alternatives.map((row) => row.id));
    await page.waitForFunction((id) => document.querySelector("h1")?.textContent === id.replace("episode_", "Episode "), alternatives[2].id);

    if (runId === "pair-01") {
      await page.locator('[data-episode="episode_0053"]').click();
      await page.waitForSelector(".evaluation-report .notice");
      assert.match(await page.locator(".evaluation-report").first().innerText(), /현재 원본이 다릅니다/);
      const firstScores = await page.locator(".scores tbody tr").first().locator("td").allTextContents();
      assert.deepEqual(firstScores.slice(0, 2), ["—", "—"]);
    }
    const hasDuplex = await page.locator('#run-select option[value="duplex"]').count();
    if (hasDuplex) {
      await page.locator("#run-select").selectOption("duplex");
      await page.waitForSelector(".utterance");
      await page.waitForFunction(() => document.querySelector("audio")?.readyState >= 1);
      await page.locator("audio").evaluate((audio) => {
        audio.currentTime = 2.5;
        audio.dispatchEvent(new Event("timeupdate"));
      });
      assert.equal(await page.locator(".utterance.is-playing").count(), 2);
      await page.locator('[data-episode="episode_0002"]').click();
      await page.waitForFunction(() => document.querySelector("h1")?.textContent === "Episode 0002");
      assert.equal(await page.locator("audio").count(), 0);
      assert.match(await page.locator(".title-row").innerText(), /대기/);
    }

    await page.locator("#run-select").selectOption(runId);
    await page.locator('[data-episode="' + current + '"]').click();
    await page.waitForSelector("audio");
    await page.setViewportSize({ width: 390, height: 844 });
    await page.evaluate(() => window.scrollTo(0, 0));
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    if (process.env.DEMO_SCREENSHOT_PREFIX) {
      await page.screenshot({ path: process.env.DEMO_SCREENSHOT_PREFIX + "-mobile.png", fullPage: true });
    }
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({
      result: "passed", episodes: listed.episodes.length, evaluators: detail.evaluations.length,
      playback: true, seek: true, profileToggles: true, reports: true,
      switchStopsAudio: true, rapidSelection: true, mobile: true,
      duplexOverlap: Boolean(hasDuplex), browserErrors: errors.length,
    }));
  } finally {
    await browser.close();
  }
}

main().catch((error) => { console.error(error); process.exit(1); });
