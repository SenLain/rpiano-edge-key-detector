using Azure.Messaging.EventHubs;
using Azure.Storage.Blobs;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Extensions.Logging;
using System;
using System.Globalization;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using static System.Net.WebRequestMethods;

namespace PianoCloudFunction;
public class Function1
{
    private static readonly HttpClient Http = new();
    private readonly ILogger<Function1> _logger;

    public Function1(ILogger<Function1> logger)
    {
        _logger = logger;
    }

    [Function(nameof(Function1))]
    public async Task Run([EventHubTrigger("iothub-ehub-rpianohub-55846779-cc133556c7", Connection = "IotHubConnection")] EventData[] events)
    {
        foreach (EventData @event in events)
        {
            _logger.LogInformation("Event Body: {body}", @event.Body);
            _logger.LogInformation("Event Content-Type: {contentType}", @event.ContentType);

            string eventBody = Encoding.UTF8.GetString(@event.Body.ToArray());

            using JsonDocument doc = JsonDocument.Parse(eventBody);
            JsonElement root = doc.RootElement;

            if (!root.TryGetProperty("session_id", out JsonElement sessionElement))
                continue;

            string sessionId = sessionElement.GetString() ?? string.Empty;
            string timestamp = root.GetProperty("timestamp").GetString() ?? DateTime.UtcNow.ToString("o");
            string deviceId = root.GetProperty("device_id").GetString() ?? string.Empty;
            int totalNotes = root.GetProperty("total_notes").GetInt32();

            string predictedKey = await PredictMusicalKeyAsync(root);
            await AppendSessionToBlobAsync(timestamp, sessionId, deviceId, totalNotes, predictedKey);

        }
    }

    private static async Task<string> PredictMusicalKeyAsync(JsonElement root)
    {
        string? endpointUrl = Environment.GetEnvironmentVariable("AutoMlUrl");
        string? apiKey = Environment.GetEnvironmentVariable("AutoMlKey");

        // Only for testing with blank endpoint url
        if (string.IsNullOrWhiteSpace(endpointUrl))
            return "8B - C Major (Mock)";

        var payload = new
        {
            input_data = new
            {
                columns = new[]
                {
                    "pct_C", "pct_Cs", "pct_D", "pct_Ds", "pct_E", "pct_F",
                    "pct_Fs", "pct_G", "pct_Gs", "pct_A", "pct_As", "pct_B",
                    "first_pc", "last_pc"
                },
                data = new[]
                {
                    new object[]
                    {
                        root.GetProperty("pct_C").GetDouble(),
                        root.GetProperty("pct_C#").GetDouble(),
                        root.GetProperty("pct_D").GetDouble(),
                        root.GetProperty("pct_D#").GetDouble(),
                        root.GetProperty("pct_E").GetDouble(),
                        root.GetProperty("pct_F").GetDouble(),
                        root.GetProperty("pct_F#").GetDouble(),
                        root.GetProperty("pct_G").GetDouble(),
                        root.GetProperty("pct_G#").GetDouble(),
                        root.GetProperty("pct_A").GetDouble(),
                        root.GetProperty("pct_A#").GetDouble(),
                        root.GetProperty("pct_B").GetDouble(),
                        root.GetProperty("first_pc").GetInt32(),
                        root.GetProperty("last_pc").GetInt32()
                    }
                }
            }
        };

        using var request = new HttpRequestMessage(HttpMethod.Post, endpointUrl);
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);
        request.Content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json");

        using HttpResponseMessage response = await Http.SendAsync(request);
        response.EnsureSuccessStatusCode();

        // Default AutoML endpoint returns e.g. ["8B - C Major"]
        string rawResult = await response.Content.ReadAsStringAsync();
        return rawResult.Trim('[', ']', '"', ' ');
    }

    private static async Task AppendSessionToBlobAsync(
        string timestamp, string sessionId, string deviceId, int totalNotes, string predictedKey)
    {
        string storageConn = Environment.GetEnvironmentVariable("AzureWebJobsStorage")!;
        var containerClient = new BlobContainerClient(storageConn, "piano-sessions");
        await containerClient.CreateIfNotExistsAsync();

        var blobClient = containerClient.GetBlobClient("sessions.csv");
        string csvContent = "timestamp,session_id,device_id,total_notes,predicted_key\n";

        if (await blobClient.ExistsAsync())
        {
            var download = await blobClient.DownloadContentAsync();
            csvContent = download.Value.Content.ToString();
        }

        string line = string.Format(
            CultureInfo.InvariantCulture,
            "{0},{1},{2},{3},\"{4}\"\n",
            timestamp, sessionId, deviceId, totalNotes, predictedKey);

        csvContent += line;
        await blobClient.UploadAsync(BinaryData.FromString(csvContent), overwrite: true);
    }
}