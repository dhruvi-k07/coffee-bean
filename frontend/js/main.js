AOS.init({
  duration: 800,
  easing: "slide",
});

(function ($) {
  "use strict";

  $(window).stellar({
    responsive: true,
    parallaxBackgrounds: true,
    parallaxElements: true,
    horizontalScrolling: false,
    hideDistantElements: false,
    scrollProperty: "scroll",
    horizontalOffset: 0,
    verticalOffset: 0,
  });

  // Scrollax
  $.Scrollax();

  var fullHeight = function () {
    $(".js-fullheight").css("height", $(window).height());
    $(window).resize(function () {
      $(".js-fullheight").css("height", $(window).height());
    });
  };
  fullHeight();

  // loader
  var loader = function () {
    setTimeout(function () {
      if ($("#ftco-loader").length > 0) {
        $("#ftco-loader").removeClass("show");
      }
    }, 1);
  };
  loader();

  // Scrollax
  $.Scrollax();

  var carousel = function () {
    $(".home-slider").owlCarousel({
      loop: true,
      autoplay: true,
      margin: 0,
      animateOut: "fadeOut",
      animateIn: "fadeIn",
      nav: false,
      autoplayHoverPause: false,
      items: 1,
      navText: [
        "<span class='ion-md-arrow-back'></span>",
        "<span class='ion-chevron-right'></span>",
      ],
      responsive: {
        0: {
          items: 1,
          nav: false,
        },
        600: {
          items: 1,
          nav: false,
        },
        1000: {
          items: 1,
          nav: false,
        },
      },
    });
    $(".carousel-work").owlCarousel({
      autoplay: true,
      center: true,
      loop: true,
      items: 1,
      margin: 30,
      stagePadding: 0,
      nav: true,
      navText: [
        '<span class="ion-ios-arrow-back">',
        '<span class="ion-ios-arrow-forward">',
      ],
      responsive: {
        0: {
          items: 1,
          stagePadding: 0,
        },
        600: {
          items: 2,
          stagePadding: 50,
        },
        1000: {
          items: 3,
          stagePadding: 100,
        },
      },
    });
  };
  carousel();

  $("nav .dropdown").hover(
    function () {
      var $this = $(this);
      // 	 timer;
      // clearTimeout(timer);
      $this.addClass("show");
      $this.find("> a").attr("aria-expanded", true);
      // $this.find('.dropdown-menu').addClass('animated-fast fadeInUp show');
      $this.find(".dropdown-menu").addClass("show");
    },
    function () {
      var $this = $(this);
      // timer;
      // timer = setTimeout(function(){
      $this.removeClass("show");
      $this.find("> a").attr("aria-expanded", false);
      // $this.find('.dropdown-menu').removeClass('animated-fast fadeInUp show');
      $this.find(".dropdown-menu").removeClass("show");
      // }, 100);
    }
  );

  $("#dropdown04").on("show.bs.dropdown", function () {
    console.log("show");
  });

  // scroll
  var scrollWindow = function () {
    $(window).scroll(function () {
      var $w = $(this),
        st = $w.scrollTop(),
        navbar = $(".ftco_navbar"),
        sd = $(".js-scroll-wrap");

      if (st > 150) {
        if (!navbar.hasClass("scrolled")) {
          navbar.addClass("scrolled");
        }
      }
      if (st < 150) {
        if (navbar.hasClass("scrolled")) {
          navbar.removeClass("scrolled sleep");
        }
      }
      if (st > 350) {
        if (!navbar.hasClass("awake")) {
          navbar.addClass("awake");
        }

        if (sd.length > 0) {
          sd.addClass("sleep");
        }
      }
      if (st < 350) {
        if (navbar.hasClass("awake")) {
          navbar.removeClass("awake");
          navbar.addClass("sleep");
        }
        if (sd.length > 0) {
          sd.removeClass("sleep");
        }
      }
    });
  };
  scrollWindow();

  var counter = function () {
    $("#section-counter").waypoint(
      function (direction) {
        if (
          direction === "down" &&
          !$(this.element).hasClass("ftco-animated")
        ) {
          var comma_separator_number_step =
            $.animateNumber.numberStepFactories.separator(",");
          $(".number").each(function () {
            var $this = $(this),
              num = $this.data("number");
            console.log(num);
            $this.animateNumber(
              {
                number: num,
                numberStep: comma_separator_number_step,
              },
              7000
            );
          });
        }
      },
      { offset: "95%" }
    );
  };
  counter();

  var contentWayPoint = function () {
    var i = 0;
    $(".ftco-animate").waypoint(
      function (direction) {
        if (
          direction === "down" &&
          !$(this.element).hasClass("ftco-animated")
        ) {
          i++;

          $(this.element).addClass("item-animate");
          setTimeout(function () {
            $("body .ftco-animate.item-animate").each(function (k) {
              var el = $(this);
              setTimeout(
                function () {
                  var effect = el.data("animate-effect");
                  if (effect === "fadeIn") {
                    el.addClass("fadeIn ftco-animated");
                  } else if (effect === "fadeInLeft") {
                    el.addClass("fadeInLeft ftco-animated");
                  } else if (effect === "fadeInRight") {
                    el.addClass("fadeInRight ftco-animated");
                  } else {
                    el.addClass("fadeInUp ftco-animated");
                  }
                  el.removeClass("item-animate");
                },
                k * 50,
                "easeInOutExpo"
              );
            });
          }, 100);
        }
      },
      { offset: "95%" }
    );
  };
  contentWayPoint();

  // navigation
  var OnePageNav = function () {
    $(".smoothscroll[href^='#'], #ftco-nav ul li a[href^='#']").on(
      "click",
      function (e) {
        e.preventDefault();

        var hash = this.hash,
          navToggler = $(".navbar-toggler");
        $("html, body").animate(
          {
            scrollTop: $(hash).offset().top,
          },
          700,
          "easeInOutExpo",
          function () {
            window.location.hash = hash;
          }
        );

        if (navToggler.is(":visible")) {
          navToggler.click();
        }
      }
    );
    $("body").on("activate.bs.scrollspy", function () {
      console.log("nice");
    });
  };
  OnePageNav();

  // magnific popup
  $(".image-popup").magnificPopup({
    type: "image",
    closeOnContentClick: true,
    closeBtnInside: true,
    fixedContentPos: true,
    mainClass: "mfp-no-margins mfp-with-zoom", // class to remove default margin from left and right side
    gallery: {
      enabled: true,
      navigateByImgClick: true,
      preload: [0, 1], // Will preload 0 - before current, and 1 after the current image
    },
    image: {
      verticalFit: true,
    },
    zoom: {
      enabled: true,
      duration: 300, // don't foget to change the duration also in CSS
    },
  });

  $(".popup-youtube, .popup-vimeo, .popup-gmaps").magnificPopup({
    disableOn: 700,
    type: "iframe",
    mainClass: "mfp-fade",
    removalDelay: 160,
    preloader: false,

    fixedContentPos: false,
  });

  $(".appointment_date").datepicker({
    format: "m/d/yyyy",
    autoclose: true,
  });

  $(".appointment_time").timepicker();

  document
    .getElementById("imageInput")
    .addEventListener("change", function (event) {
      var reader = new FileReader();

      reader.onload = function () {
        var output = document.getElementById("imagePreview");
        output.src = reader.result; // Set the src of the img tag to the read file

        // Set fixed height and width
        output.style.width = "200px"; // Adjust as needed
        output.style.height = "200px"; // Adjust as needed
      };

      reader.readAsDataURL(event.target.files[0]); // Read the file as a Data URL (base64)
    });

  document
    .getElementById("uploadForm")
    .addEventListener("submit", function (e) {
      e.preventDefault(); // Stop the form from submitting normally

      const input = document.getElementById("imageInput");
      const file = input.files[0]; // Get the file
      if (!file) {
        alert("Please select a file.");
        return;
      }

      const formData = new FormData();
      formData.append("file", file); // Match the key to your Flask endpoint's expected key

      // Modify the URL to match your Flask application's URL and endpoint
      fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.json()) // Assuming the response is text
        .then((data) => {
          // document.getElementById('message').textContent = `Roasting Level of selected bean is ${data.prediction}!`;
          let predictionMessage = `Roasting Level of selected bean: `;

		  let tipsTricks = '';

		  // Determine which set of tips to display based on the prediction
		  switch (data.prediction) {
			case "Dark":
				tipsTricks = `<ul>
					<li><strong>Flavor Profile: </strong>Dark roast beans typically have a bold, robust flavor with a heavy body and less acidity. The roasting process brings out a rich, chocolatey, or smoky taste, often with hints of caramel.</li>
					<li><strong>Brewing Tips: </strong>Use a lower water temperature (around 195°F to 205°F) to avoid over-extraction, which can make the coffee bitter. French press or espresso methods highlight the rich and bold flavors of dark roasts.</li>
					<li><strong>Grinding: </strong>A coarser grind works well, especially for French press or cold brew, to balance extraction and prevent bitterness</li>
				</ul>`;
				break;
			case "Medium":
				tipsTricks = `<ul>
					<li><strong>Flavor Profile: </strong>Medium roast offers a balanced flavor, aroma, and acidity. These beans tend to have a fuller body than light roasts, with more complex flavors that can include fruity, nutty, or spicy notes.</li>
					<li><strong>Brewing Tips: </strong>Medium roast beans are versatile and can be used with a variety of brewing methods, including drip, pour-over, and espresso. Aim for a water temperature of about 195°F to 205°F.</li>
					<li><strong>Grinding: </strong>A medium grind is generally best, as it suits a wide range of brewing methods, from pour-over to drip coffee makers.</li>
				</ul>`;
				break;
			case "Light":
				tipsTricks = `<ul>
					<li><strong>Flavor Profile: </strong>Light roast beans retain the most caffeine and original flavor of the coffee bean, including floral, fruity, and acidic notes. They offer a lighter body and no oil on the surface of the beans.</li>
					<li><strong>Brewing Tips: </strong>Light roasts do well with methods that allow their delicate flavors to shine, such as pour-over or Aeropress. Use slightly hotter water (205°F to 210°F) to ensure proper extraction.</li>
					<li><strong>Grinding: </strong>A finer grind can help extract the full range of flavors from light roast beans, especially in methods like espresso or Aeropress.</li>
				</ul>`;
				break;
			case "Green":
				tipsTricks = `<ul>
					<li><strong>Storage: </strong>Green beans should be stored in a cool, dark place. Proper storage can keep them fresh for months, even up to a year.</li>
					<li><strong>Roasting: </strong>Roasting green beans at home allows you to customize the roast level to your preference. Use a home roasting machine, oven, or even a popcorn popper. Start with a small batch to experiment.</li>
					<li><strong>Roasting Tips: </strong>Monitor the beans closely during roasting. Listen for the "first crack" and "second crack" sounds to gauge the roast level. Cooling the beans immediately after roasting is crucial to stop the cooking process.</li>
					<li><strong>Freshness: </strong>Use freshly roasted beans within two weeks for the best flavor, and let them degas for at least 24 hours before brewing.</li>
				</ul>`;
				break;
			default:
				tipsTricks = "<p>No specific tips available for this roast level.</p>";
		}

          // Create a new div element to hold the enriched message
          let messageDiv = document.createElement("div");

          // Set the inner HTML of the message div
          messageDiv.innerHTML = `
		  <h2 class="mb-4">Result</h2>
		  <div class="container"> 
			  <div class="row">
					<div class="col-md-12 align-items-center"> 
					  <div class="prediction-message">${predictionMessage}<strong>${data.prediction}</strong>
					  </div>
				  </div>
				  <div class="col-md-12 prediction-descrip align-items-left">
						  ${tipsTricks}
					</div>
			  </div>
		  </div>`;

          // Finally, clear any existing content and append the newly created message div to your target container
          let messageContainer = document.getElementById("message");
          messageContainer.innerHTML = ""; // Clear existing content if any
          messageContainer.appendChild(messageDiv);
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    });
})(jQuery);
