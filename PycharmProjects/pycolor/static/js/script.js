var interval;
var progress = 0;

$(function() {
    $('button').click(function() {
        var user = $('#txtUsername').val();
        var pass = $('#txtPassword').val();
        interval = setInterval(update_progress, 1000);
        $.ajax({
            url: '/analysis',
            data: $('form').serialize(),
            type: 'POST',
            success: function(response) {
                $('.progress-bar').animate({'width': "100%"}).attr('aria-valuenow', "100%");
                $('#resultat').html('<img src="' + response + '" />');
            },
            error: function(error) {
                console.log(error);
            }
        });
    });
});

function update_progress() {
     $.get('/progress').done(function(n){
         if (n >= 100) {
            clearInterval(interval);
         } else {
            progress = n;
            $('.progress-bar').animate({'width': n +'%'}).attr('aria-valuenow', n);
         }
     }).fail(function() {
         clearInterval(interval);
          console.log("Error from server connexion.");
     });
}